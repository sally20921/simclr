import os
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import onnx
import onnxruntime as ort

from datasets.datasets import Datasets
from utils.logger import setup_logger
from config.simclr_config import SimCLRConfig
from models.simclr import SimCLR


def setup_parser():
    parser = argparse.ArgumentParser(description="Train a cnn for a classification task using contrastive learning")
    parser.add_argument("config", help="path to config file")
    parser.add_argument("model", help="path to model file")
    parser.add_argument("epoch_num", help="epoch number")
    return parser


def save_onnx_model(torch_model, num_classes, config, current_epoch):
    logger = logging.getLogger(config.base.logger_name)
    onnx_model_file_path = os.path.join(config.base.log_dir_path, "checkpoint_{}.pth".format(current_epoch))

    # input to the model
    x = torch.randn(config.fine_tuning.batch_size, 3, config.fine_tuning.img_size,
                    config.fine_tuning.img_size, requires_grad=True)
    logger.info('created random onnx model input')

    torch_out = torch_model(x)
    logger.info('tested random onnx model input')

    # export the model
    torch.onnx.export(
        # model being run
        torch_model,
        # model input (or a tuple for multiple inputs)
        x,
        # where to save the model (can be a file or file-like object)
        onnx_model_file_path,
        # store the trained parameter weights inside the model file
        export_params=True,
        # the ONNX version to export the model to
        opset_version=10,
        # whether to execute constant folding for optimization
        do_constant_folding=True,
        # the models input names
        input_names=['input'],
        # the model's output names
        output_names=['output'],
        # variable length axes
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}})

    logger.info('saved onnx model: {}'.format(onnx_model_file_path))

    return onnx_model_file_path


def load_onnx_model(config, onnx_model_file_path):
    # load the ONNX model
    onnx_model = onnx.load(onnx_model_file_path)

    # check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)

    return onnx_model


def load_torch_model(config, num_classes):
    logger = logging.getLogger(config.base.logger_name)

    torch_model = SimCLR.get_resnet_model(config.simclr.model.resnet)
    num_ftrs = torch_model.fc.in_features
    torch_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    logger.info('loaded classification model')

    model_file_path = os.path.join(config.onnx.model_path, "checkpoint_{}.pth".format(config.onnx.epoch_num))
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('invalid model_file_path: {}'.format(model_file_path))

    state_dict = torch.load(model_file_path, map_location=config.base.device.type)
    torch_model.load_state_dict(state_dict)
    logger.info('loaded state dict into classification model')

    return torch_model


def test_onnx_model(config, onnx_model_file_path, val_dataset, test_dataset, val_loader, test_loader):
    logger = logging.getLogger(config.base.logger_name)
    logger.info('using onnx_model_file_path: {}'.format(onnx_model_file_path))

    num_test_images = len(test_dataset)
    logger.info('using test_loader with num_images: {}'.format(num_test_images))

    num_val_images = len(val_dataset)
    logger.info('using val_loader with num_images: {}'.format(num_val_images))

    sess = ort.InferenceSession(onnx_model_file_path)
    input_name = sess.get_inputs()[0].name

    running_corrects = 0

    for step, (batch, results) in enumerate(test_loader):

        input = batch.detach().cpu().numpy() if batch.requires_grad else batch.cpu().numpy()

        outputs_list = sess.run(None, {input_name: input})[0]

        outputs_tensor = torch.FloatTensor(outputs_list)

        _, preds_onnx = torch.max(outputs_tensor, 1)

        step_corrects = torch.sum(preds_onnx == results.data)
        running_corrects += step_corrects

        if step % 1 == 0:
            logger.info("step [%5.i|%5.i] -> step_corrects: %i, running_corrects: %i" % (
                step + 1, len(test_loader), step_corrects, running_corrects))

    test_acc = running_corrects.double() / num_test_images
    logger.info('test acc: {}'.format(test_acc))

    running_corrects = 0

    for step, (batch, results) in enumerate(val_loader):

        input = batch.detach().cpu().numpy() if batch.requires_grad else batch.cpu().numpy()

        outputs_list = sess.run(None, {input_name: input})[0]

        outputs_tensor = torch.FloatTensor(outputs_list)

        _, preds_onnx = torch.max(outputs_tensor, 1)

        step_corrects = torch.sum(preds_onnx == results.data)
        running_corrects += step_corrects

        if step % 1 == 0:
            logger.info("step [%5.i|%5.i] -> step_corrects: %i, running_corrects: %i" % (
                step + 1, len(val_loader), step_corrects, running_corrects))

    val_acc = running_corrects.double() / num_val_images
    logger.info('val acc: {}'.format(val_acc))

    return val_acc, test_acc


def test_pt_model(config, pt_model, val_dataset, test_dataset, val_loader, test_loader):
    logger = logging.getLogger(config.base.logger_name)

    num_val_images = len(val_dataset)
    logger.info('using val_loader with num_images: {}'.format(num_val_images))

    num_test_images = len(test_dataset)
    logger.info('using test_loader with num_images: {}'.format(num_test_images))

    pt_model = pt_model.to(config.base.device)
    pt_model.eval()

    running_corrects = 0

    for step, (batch, results) in enumerate(val_loader):

        inputs = batch.to(config.base.device)
        labels = results.to(config.base.device)

        outputs = pt_model(inputs)
        _, preds = torch.max(outputs, 1)

        step_corrects = torch.sum(preds == labels.data)
        running_corrects += step_corrects

        if step % 1 == 0:
            logger.info("step [%5.i|%5.i] -> step_corrects: %i, running_corrects: %i" % (
                step + 1, len(test_loader), step_corrects, running_corrects))

    val_acc = running_corrects.double() / num_test_images
    logger.info('val acc: {}'.format(val_acc))

    running_corrects = 0

    for step, (batch, results) in enumerate(test_loader):

        inputs = batch.to(config.base.device)
        labels = results.to(config.base.device)

        outputs = pt_model(inputs)
        _, preds = torch.max(outputs, 1)

        step_corrects = torch.sum(preds == labels.data)
        running_corrects += step_corrects

        if step % 1 == 0:
            logger.info("step [%5.i|%5.i] -> step_corrects: %i, running_corrects: %i" % (
                step + 1, len(test_loader), step_corrects, running_corrects))

    test_acc = running_corrects.double() / num_val_images
    logger.info('test acc: {}'.format(test_acc))

    return val_acc, test_acc


def main(args):
    config_yaml = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if not os.path.exists(args.config):
        raise FileNotFoundError('provided config file does not exist: %s' % args.config)

    config_yaml['logger_name'] = 'onnx'
    config = SimCLRConfig(config_yaml)

    if not os.path.exists(config.base.output_dir_path):
        os.mkdir(config.base.output_dir_path)

    if not os.path.exists(config.base.log_dir_path):
        os.makedirs(config.base.log_dir_path)

    logger = setup_logger(config.base.logger_name, config.base.log_file_path)
    logger.info('using config: %s' % config)

    if not os.path.exists(args.model):
        raise FileNotFoundError('provided model directory does not exist: %s' % args.model)
    else:
        logger.info('using model directory: %s' % args.model)

    config.onnx.model_path = args.model
    logger.info('using model_path: {}'.format(config.onnx.model_path))

    config.onnx.epoch_num = args.epoch_num
    logger.info('using epoch_num: {}'.format(config.onnx.epoch_num))

    model_file_path = Path(config.onnx.model_path).joinpath(
        'checkpoint_' + config.onnx.epoch_num + '.pth')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError('model file does not exist: %s' % model_file_path)
    else:
        logger.info('using model file: %s' % model_file_path)

    train_dataset, val_dataset, test_dataset, classes = Datasets.get_datasets(config)
    num_classes = len(classes)

    train_loader, val_loader, test_loader = Datasets.get_loaders(config, train_dataset, val_dataset, test_dataset)

    torch_model = load_torch_model(config, num_classes)

    val_acc, test_acc = test_pt_model(config, torch_model, val_dataset, test_dataset, val_loader, test_loader)
    logger.info('torch model performance -> val_acc: {}, test_acc: {}'.format(val_acc, test_acc))

    torch_model = torch_model.to(torch.device('cpu'))
    onnx_model_file_path = save_onnx_model(torch_model, num_classes=num_classes, config=config,
                                           current_epoch=config.onnx.epoch_num)

    onnx_model = load_onnx_model(config, onnx_model_file_path)
    if onnx_model:
        logger.info('loaded onnx_model: {}'.format(onnx_model_file_path))

    val_acc, test_acc = test_onnx_model(config, onnx_model_file_path, val_dataset, test_dataset, val_loader,
                                        test_loader)
    logger.info('onnx model performance -> val_acc: {}, test_acc: {}'.format(val_acc, test_acc))


if __name__ == '__main__':
    np.random.seed(0)
    main(setup_parser().parse_args())
