import os
import logging
import argparse
from pathlib import Path
import shutil
import copy

import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import Datasets
from utils.logger import setup_logger
from config.simclr_config import SimCLRConfig
from models.simclr import SimCLR
from models.logistic_regression import LogisticRegression


def setup_parser():
    parser = argparse.ArgumentParser(description="Train a logistic regression model based on a pretraied simclr model")
    parser.add_argument("config", help="path to config file")
    parser.add_argument("model", help="path to simclr model file")
    parser.add_argument("epoch_num", help="epoch number of the pretrained simclr model")
    parser.add_argument("--data_dir_path", help="data_dir_path to be used instead of one in the config file")
    return parser


def load_simclr_model(config):
    model = SimCLR(config)

    model_file_path = os.path.join(config.logistic_regression.model_path,
                                   "checkpoint_{}.pth".format(config.logistic_regression.epoch_num))

    model.load_state_dict(torch.load(model_file_path, map_location=config.base.device.type))
    model = model.to(config.base.device)

    return model


def compute_features(config, loader, context_model):
    logger = logging.getLogger(config.base.logger_name)

    features = []
    labels = []

    log_every_n_steps = 10

    for step, (x, y) in enumerate(loader):
        x = x.to(config.base.device)

        with torch.no_grad():
            h, z = context_model(x)

        h = h.detach()

        features.extend(h.cpu().detach().numpy())
        labels.extend(y.numpy())

        if step % log_every_n_steps == 0:
            logger.info("step [{:>3}|{:>3}] -> computing features".format(step + 1, len(loader)))

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def get_features(config, simclr_model, train_loader, test_loader):
    logger = logging.getLogger(config.base.logger_name)

    train_x, train_y = compute_features(config, train_loader, simclr_model)
    logger.info('computed train features')

    test_x, test_y = compute_features(config, test_loader, simclr_model)
    logger.info('computed test features')

    return train_x, train_y, test_x, test_y


def get_data_loaders(config, x_train, y_train, x_test, y_test):
    logger = logging.getLogger(config.base.logger_name)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.logistic_regression.batch_size,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.logistic_regression.batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def train(config, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0

    for step, (images, labels) in enumerate(loader):
        optimizer.zero_grad()

        images = images.to(config.base.device)
        labels = labels.to(config.base.device)

        output = model(images)
        loss = criterion(output, labels)

        predicted = output.argmax(1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(config, loader, model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()

    for step, (images, labels) in enumerate(loader):
        model.zero_grad()

        images = images.to(config.base.device)
        labels = labels.to(config.base.device)

        output = model(images)
        loss = criterion(output, labels)

        predicted = output.argmax(1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def main(args):
    config_yaml = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if not os.path.exists(args.config):
        raise FileNotFoundError('provided config file does not exist: %s' % args.config)

    if 'restart_log_dir_path' not in config_yaml['simclr']['train'].keys():
        config_yaml['simclr']['train']['restart_log_dir_path'] = None

    if args.data_dir_path is not None:
        config_yaml['simclr']['train']['data_dir_path'] = args.data_dir_path
        print('yo!: ', args.data_dir_path)

    config_yaml['logger_name'] = 'logreg'
    config = SimCLRConfig(config_yaml)

    if not os.path.exists(config.base.output_dir_path):
        os.mkdir(config.base.output_dir_path)

    if not os.path.exists(config.base.log_dir_path):
        os.makedirs(config.base.log_dir_path)

    logger = setup_logger(config.base.logger_name, config.base.log_file_path)
    logger.info('using config: %s' % config)

    config_copy_file_path = os.path.join(config.base.log_dir_path, 'config.yaml')
    shutil.copy(args.config, config_copy_file_path)

    writer = SummaryWriter(log_dir=config.base.log_dir_path)

    if not os.path.exists(args.model):
        raise FileNotFoundError('provided model directory does not exist: %s' % args.model)
    else:
        logger.info('using model directory: %s' % args.model)

    config.logistic_regression.model_path = args.model
    logger.info('using model_path: {}'.format(config.logistic_regression.model_path))

    config.logistic_regression.epoch_num = args.epoch_num
    logger.info('using epoch_num: {}'.format(config.logistic_regression.epoch_num))

    model_file_path = Path(config.logistic_regression.model_path).joinpath(
        'checkpoint_' + config.logistic_regression.epoch_num + '.pth')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('model file does not exist: %s' % model_file_path)
    else:
        logger.info('using model file: %s' % model_file_path)

    train_dataset, val_dataset, test_dataset, classes = Datasets.get_datasets(config,
                                                                              img_size=config.logistic_regression.img_size)
    num_classes = len(classes)

    train_loader, val_loader, test_loader = Datasets.get_loaders(config, train_dataset, val_dataset, test_dataset)

    simclr_model = load_simclr_model(config)
    simclr_model = simclr_model.to(config.base.device)
    simclr_model.eval()

    model = LogisticRegression(simclr_model.num_features, num_classes)
    model = model.to(config.base.device)

    learning_rate = config.logistic_regression.learning_rate
    momentum = config.logistic_regression.momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("creating features from pre-trained context model")
    (train_x, train_y, test_x, test_y) = get_features(
        config, simclr_model, train_loader, test_loader
    )

    feature_train_loader, feature_test_loader = get_data_loaders(
        config, train_x, train_y, test_x, test_y
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = 0

    for epoch in range(config.logistic_regression.epochs):
        loss_epoch, accuracy_epoch = train(
            config, feature_train_loader, model, criterion, optimizer
        )

        loss = loss_epoch / len(train_loader)
        accuracy = accuracy_epoch / len(train_loader)

        writer.add_scalar("Loss/train_epoch", loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", accuracy, epoch)
        logger.info(
            "epoch [%3.i|%i] -> train loss: %f, accuracy: %f" % (
                epoch + 1, config.logistic_regression.epochs, loss, accuracy)
        )

        if accuracy > best_acc:
            best_loss = loss
            best_epoch = epoch + 1
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    logger.info(
        "train dataset performance -> best epoch: {}, loss: {}, accuracy: {}".format(best_epoch, best_loss, best_acc, )
    )

    loss_epoch, accuracy_epoch = test(
        config, feature_test_loader, model, criterion
    )

    loss = loss_epoch / len(test_loader)
    accuracy = accuracy_epoch / len(test_loader)
    logger.info(
        "test dataset performance -> best epoch: {}, loss: {}, accuracy: {}".format(best_epoch, loss, accuracy)
    )


if __name__ == '__main__':
    main(setup_parser().parse_args())
