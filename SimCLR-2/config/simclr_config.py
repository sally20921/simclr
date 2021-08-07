import os
from datetime import datetime
import torch

from dataclasses import dataclass


class SimCLRConfig:
    @dataclass()
    class Base:
        output_dir_path: str
        log_dir_path: str
        log_file_path: str
        device: object
        num_gpu: int
        logger_name: str

    @dataclass()
    class Train:
        # batch_size as usual. examples: 16,32,..
        batch_size: int

        # number of workers to be used for data loading. examples: 2,4,...
        num_workers: int

        # start training with this epoch. most likely: 0
        start_epoch: int

        # in case of restart this is where the saved model is expected to be located
        restart_log_dir_path: str

        # end training with this epoch. examples: 10, 100,...
        epochs: int

        # directory where the datasets are located. example: "/home/USER_NAME/Data"
        data_dir_path: str

        # dataset name. options: ["CIFAR10", "STL10", "iNaturalist2019", "ImageNet"]
        dataset: str

        # save trained model every n epochs. examples: 1,5,10,...
        save_num_epochs: int

        # image size obtained from last data preparation step
        img_size: int

        # name of the optimizer. options: ["Adam", "LARS"]
        # TODO: implement LARS ptimizer
        optimizer: str

        weight_decay: float

        temperature: float

        global_step: int

        current_epoch: int

    @dataclass()
    class Model:
        # model architecture. options: ["resnet18", "resnet50"]
        resnet: str
        normalize: bool
        projection_dim: int

    @dataclass()
    class SimCLR:
        train: object
        model: object

    @dataclass()
    class LogisticRegression:
        epochs: int
        batch_size: int
        learning_rate: float
        momentum: float
        img_size: int

        model_path: str
        epoch_num: int

    @dataclass()
    class FineTuning:
        epochs: int
        batch_size: int
        learning_rate: float
        momentum: float
        img_size: int
        save_num_epochs: int

        # decay "learning_rate" by a factor of "gamma" every "step_size" epochs
        gamma: float
        step_size: int

        model_path: str
        epoch_num: int

    @dataclass()
    class ONNX:
        batch_size: int
        img_size: int
        model_path: str
        epoch_num: int

    def __init__(self, config):
        global_step = 0
        current_epoch = 0

        simclr_train = SimCLRConfig.Train(**config['simclr']['train'], global_step=global_step,
                                          current_epoch=current_epoch)
        simclr_model = SimCLRConfig.Model(**config['simclr']['model'])
        self.simclr = SimCLRConfig.SimCLR(simclr_train, simclr_model)

        model_path = None
        epoch_num = None
        self.logistic_regression = SimCLRConfig.LogisticRegression(**config['logistic_regression'],
                                                                   model_path=model_path, epoch_num=epoch_num)
        model_path = None
        epoch_num = None
        self.fine_tuning = SimCLRConfig.FineTuning(**config['fine_tuning'], model_path=model_path,
                                                   epoch_num=epoch_num)

        model_path = None
        epoch_num = None
        self.onnx = SimCLRConfig.ONNX(**config['onnx'], model_path=model_path, epoch_num=epoch_num)

        logger_name = config['logger_name']

        output_dir_path = 'output'
        now = datetime.now()
        dt_string: str = now.strftime("%Y_%m_%d_%H_%M_%S")
        log_dir_name = dt_string + '_' + logger_name + '_' + self.simclr.train.dataset.lower()

        log_dir_path = os.path.join(output_dir_path, log_dir_name)
        log_file_path = os.path.join(log_dir_path, 'log.txt')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        num_gpu = torch.cuda.device_count()

        self.base = SimCLRConfig.Base(output_dir_path, log_dir_path, log_file_path, device, num_gpu, logger_name)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
