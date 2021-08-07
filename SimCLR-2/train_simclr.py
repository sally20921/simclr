import os
import logging
import argparse
import shutil

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import Datasets
from utils.logger import setup_logger
from config.simclr_config import SimCLRConfig
from models.simclr import SimCLR
from loss_functions.nt_xent import NTXent


def setup_parser():
    parser = argparse.ArgumentParser(description="Train a cnn for a classification task using contrastive learning")
    parser.add_argument("config", help="path to config file")
    return parser


def load_model(config):

    model = SimCLR(config)

    if not config.simclr.train.start_epoch == 0:

        model_file_path = os.path.join(config.simclr.train.restart_log_dir_path,
                                       "checkpoint_{}.pth".format(config.simclr.train.start_epoch))

        if not os.path.exists(model_file_path):
            raise FileNotFoundError('provided checkpoint does not exist: {}'.format(model_file_path))

        model.load_state_dict(torch.load(model_file_path, map_location=config.base.device.type))

    model = model.to(config.base.device)

    return model


def save_model(config, model):
    model_file_path = os.path.join(config.base.log_dir_path,
                                   "checkpoint_{}.pth".format(config.simclr.train.current_epoch))
    torch.save(model.state_dict(), model_file_path)

    return model_file_path


def train(config, train_loader, model, criterion, optimizer, writer):
    logger = logging.getLogger(config.base.logger_name)

    # TODO: this should be a config attribute
    log_every_n_steps = 100

    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(config.base.device)
        x_j = x_j.to(config.base.device)

        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)

        loss.backward()

        optimizer.step()

        if step % log_every_n_steps == 0:
            logger.info("step [%5.i|%5.i] -> loss: %.15f" % (step + 1, len(train_loader), loss.item()))

        writer.add_scalar("Loss/train", loss.item(), config.simclr.train.global_step)
        loss_epoch += loss.item()
        config.simclr.train.global_step += 1
    return loss_epoch


def main(args):
    config_yaml = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if not os.path.exists(args.config):
        raise FileNotFoundError('provided config file does not exist: {}'.format(args.config))

    if 'restart_log_dir_path' not in config_yaml['simclr']['train'].keys():
        config_yaml['simclr']['train']['restart_log_dir_path'] = None

    config_yaml['logger_name'] = 'simclr'
    config = SimCLRConfig(config_yaml)

    if not config.simclr.train.start_epoch == 0 and config.simclr.train.restart_log_dir_path is None:
        raise ValueError('provided config file is invalid. no restart_log_dir_path provided and start_epoch is not 0')

    if not os.path.exists(config.base.output_dir_path):
        os.mkdir(config.base.output_dir_path)

    if not os.path.exists(config.base.log_dir_path):
        os.makedirs(config.base.log_dir_path)

    logger = setup_logger(config.base.logger_name, config.base.log_file_path)
    logger.info('using config: {}'.format(config))

    config_copy_file_path = os.path.join(config.base.log_dir_path, 'config.yaml')
    shutil.copy(args.config, config_copy_file_path)

    writer = SummaryWriter(log_dir=config.base.log_dir_path)

    model = load_model(config)
    logger.info('loaded model')

    train_dataset, _ = Datasets.get_simclr_dataset(config)
    logger.info('using train_dataset. length: {}'.format(len(train_dataset)))

    train_loader = Datasets.get_simclr_loader(config, train_dataset)
    logger.info('created train_loader. length {}'.format(len(train_loader)))

    scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    logger.info('created optimizer')

    criterion = NTXent(config.simclr.train.batch_size, config.simclr.train.temperature, config.base.device)
    logger.info('created criterion')

    config.simclr.train.current_epoch = config.simclr.train.start_epoch
    for epoch in range(config.simclr.train.start_epoch, config.simclr.train.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(config, train_loader, model, criterion, optimizer, writer)

        if scheduler:
            scheduler.step()

        if epoch % config.simclr.train.save_num_epochs == 0:
            save_model(config, model)

        writer.add_scalar("Loss/train_epoch", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Learning rate", lr, epoch)
        logger.info(
            "epoch [%5.i|%5.i] -> loss: %.15f, lr: %f" % (
                epoch + 1, config.simclr.train.epochs, loss_epoch / len(train_loader), round(lr, 5))
        )
        config.simclr.train.current_epoch += 1

    save_model(config, model)


if __name__ == '__main__':
    main(setup_parser().parse_args())
