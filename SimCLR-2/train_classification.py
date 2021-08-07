import os
import logging
import argparse
from pathlib import Path
import shutil
import time
import copy

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import Datasets
from utils.logger import setup_logger
from config.simclr_config import SimCLRConfig
from models.simclr import SimCLR


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Train a cnn for a classification task by fine-tuning a pretrained simclr model")
    parser.add_argument("config", help="path to config file")
    parser.add_argument("model", help="path to simcrl model file")
    parser.add_argument("epoch_num", help="epoch number of the pretrained simclr model")
    return parser


def to_classification_model(simclr_model, num_classes, config):
    logger = logging.getLogger('test_transfer_learning')

    classification_model = SimCLR.get_resnet_model(config.simclr.model.resnet)
    logger.info('loaded classification model')

    num_ftrs = classification_model.fc.in_features
    classification_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    logger.info('adjusted classification model. fc layer now matches number of classes: {}'.format(num_classes))

    simclr_model.encoder.fc = torch.nn.Linear(num_ftrs, num_classes)
    logger.info('adjusted simclr model. fc layer in the encoder now matches number of classes: {}'.format(num_classes))

    state_dict = simclr_model.encoder.state_dict()
    logger.info('extracted state dict from simclr model')

    classification_model.load_state_dict(state_dict)
    logger.info('loaded state dict into classification model')

    return classification_model


def save_model(config, model, current_epoch):
    model_file_path = os.path.join(config.base.log_dir_path, "checkpoint_{}.pth".format(current_epoch))
    torch.save(model.state_dict(), model_file_path)

    return model_file_path


def load_model(config):
    model = SimCLR(config)

    model_file_path = os.path.join(config.fine_tuning.model_path,
                                   "checkpoint_{}.pth".format(config.fine_tuning.epoch_num))
    model.load_state_dict(torch.load(model_file_path, map_location=config.base.device.type))

    model = model.to(config.base.device)

    return model


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, config, num_epochs, writer):
    logger = logging.getLogger(config.base.logger_name)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    log_every_n_steps = 100

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(config.base.device)
                labels = labels.to(config.base.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if step % log_every_n_steps == 0:
                    logger.info(
                        "epoch [{:>4}|{:>4}] -> step [{:>5}|{:>5}] -> loss: {:.10}".format(epoch + 1, num_epochs,
                                                                                           step + 1,
                                                                                           len(dataloaders[phase]),
                                                                                           loss.item()))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if epoch % config.fine_tuning.save_num_epochs == 0:
                save_model(config, model, epoch)

            writer.add_scalar("Loss/{}_epoch".format(phase), epoch_loss, epoch)
            writer.add_scalar("Accuracy/{}_epoch".format(phase), epoch_acc, epoch)

            logger.info('epoch [{:>4}|{:>4}] -> {}, loss: {:.10}, accuracy: {:.10}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logger.info('training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('best validation accuracy: {:10}, epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(config, model, test_loader):
    logger = logging.getLogger(config.base.logger_name)

    model = model.to(config.base.device)
    model.eval()

    running_corrects = 0

    num_test_images = len(test_loader.sampler)

    log_every_n_steps = 100

    for step, (batch, results) in enumerate(test_loader):

        inputs = batch.to(config.base.device)
        labels = results.to(config.base.device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        step_corrects = torch.sum(preds == labels.data)
        running_corrects += step_corrects

        if step % log_every_n_steps == 0:
            logger.info("step [%5.i|%5.i] -> step_corrects: %i, running_corrects: %i" % (
                step + 1, len(test_loader), step_corrects, running_corrects))

    test_acc = running_corrects.double() / num_test_images
    logger.info('test accuracy: {}'.format(test_acc))

    return test_acc


def main(args):
    config_yaml = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if not os.path.exists(args.config):
        raise FileNotFoundError('provided config file does not exist: {}'.format(args.config))

    if 'restart_log_dir_path' not in config_yaml['simclr']['train'].keys():
        config_yaml['simclr']['train']['restart_log_dir_path'] = None

    config_yaml['logger_name'] = 'classification'
    config = SimCLRConfig(config_yaml)

    if not os.path.exists(config.base.output_dir_path):
        os.mkdir(config.base.output_dir_path)

    if not os.path.exists(config.base.log_dir_path):
        os.makedirs(config.base.log_dir_path)

    logger = setup_logger(config.base.logger_name, config.base.log_file_path)
    logger.info('using config: {}'.format(config))

    config_copy_file_path = os.path.join(config.base.log_dir_path, 'config.yaml')
    shutil.copy(args.config, config_copy_file_path)

    writer = SummaryWriter(log_dir=config.base.log_dir_path)

    if not os.path.exists(args.model):
        raise FileNotFoundError('provided model directory does not exist: %s' % args.model)
    else:
        logger.info('using model directory: {}'.format(args.model))

    config.fine_tuning.model_path = args.model
    logger.info('using model_path: {}'.format(config.fine_tuning.model_path))

    config.fine_tuning.epoch_num = args.epoch_num
    logger.info('using epoch_num: {}'.format(config.fine_tuning.epoch_num))

    model_file_path = Path(config.fine_tuning.model_path).joinpath(
        'checkpoint_' + config.fine_tuning.epoch_num + '.pth')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('model file does not exist: {}'.format(model_file_path))
    else:
        logger.info('using model file: {}'.format(model_file_path))

    train_dataset, val_dataset, test_dataset, classes = Datasets.get_datasets(config)
    num_classes = len(classes)

    train_loader, val_loader, test_loader = Datasets.get_loaders(config, train_dataset, val_dataset, test_dataset)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }

    dataset_sizes = {
        'train': len(train_loader.sampler),
        'val': len(val_loader.sampler)
    }

    simclr_model = load_model(config)
    logger.info('loaded simclr_model: {}'.format(config.fine_tuning.model_path))

    classification_model = to_classification_model(simclr_model, num_classes, config)
    classification_model = classification_model.to(config.base.device)
    logger.info('created classification model from simclr model')

    criterion = torch.nn.CrossEntropyLoss()
    logger.info('created criterion')

    lr = config.fine_tuning.learning_rate
    momentum = config.fine_tuning.momentum
    optimizer_ft = torch.optim.SGD(classification_model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    logger.info('created optimizer')

    step_size = config.fine_tuning.step_size
    gamma = config.fine_tuning.gamma
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    logger.info('created learning rate scheduler')

    epochs = config.fine_tuning.epochs
    classification_model = train_model(classification_model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                                       dataset_sizes,
                                       config,
                                       epochs,
                                       writer)
    logger.info('completed model training')

    test_model(config, classification_model, test_loader)
    logger.info('completed model testing')

    trained_model_file_path = save_model(config, classification_model, epochs)
    logger.info('saved trained model: {}'.format(trained_model_file_path))


if __name__ == '__main__':
    main(setup_parser().parse_args())
