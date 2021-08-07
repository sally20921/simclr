from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

from transformations.simclr import TransformsSimCLR
from datasets.inaturalist_2019 import INaturalist2019Dataset


class Datasets:

    @staticmethod
    def get_simclr_dataset(config):
        if config.simclr.train.dataset == "STL10":

            train_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size)
            )

            classes = train_dataset.classes

        elif config.simclr.train.dataset == "CIFAR10":

            train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size)
            )

            classes = train_dataset.classes

        elif config.simclr.train.dataset == 'ImageNet':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('ImageNet')
            train_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='train',
                                                          transform=TransformsSimCLR(size=config.simclr.train.img_size))
            classes = train_dataset.classes

        elif config.simclr.train.dataset == 'iNaturalist2019':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('iNaturalist2019')

            train_json_file_path = dataset_dir_path.joinpath('train2019_1.0.json')
            train_dataset = INaturalist2019Dataset(train_json_file_path, dataset_dir_path,
                                                   transform=TransformsSimCLR(size=config.simclr.train.img_size))

            classes = train_dataset.classes
        else:
            raise NotImplementedError('invalid dataset: {}'.format(config.simclr.train.dataset))

        return train_dataset, classes

    @staticmethod
    def get_datasets(config, img_size=None):

        if img_size is None:
            img_size = config.simclr.train.img_size

        if config.simclr.train.dataset == "STL10":

            train_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=img_size).train_test_transform,
            )

            val_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
            )

            classes = train_dataset.classes

            test_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="test",
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
            )
        elif config.simclr.train.dataset == "CIFAR10":

            train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=img_size).train_test_transform,
            )

            val_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
            )

            classes = train_dataset.classes

            test_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=False,
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
            )

        elif config.simclr.train.dataset == 'ImageNet':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('ImageNet')
            train_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='train',
                                                          transform=TransformsSimCLR(
                                                              size=img_size).train_test_transform)

            classes = train_dataset.classes

            val_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='val',
                                                        transform=TransformsSimCLR(
                                                            size=img_size).test_transform)

            # no labeled test data is available
            test_dataset = val_dataset

        elif config.simclr.train.dataset == 'iNaturalist2019':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('iNaturalist2019')

            train_json_file_path = dataset_dir_path.joinpath('train2019_1.0.json')
            train_dataset = INaturalist2019Dataset(train_json_file_path, dataset_dir_path,
                                                   transform=TransformsSimCLR(
                                                       size=img_size).train_test_transform)

            classes = train_dataset.classes

            val_json_file_path = dataset_dir_path.joinpath('val2019_1.0.json')
            val_dataset = INaturalist2019Dataset(val_json_file_path, dataset_dir_path,
                                                 transform=TransformsSimCLR(
                                                     size=img_size).test_transform)
            # no labeled test data is available
            test_dataset = val_dataset

        else:
            raise NotImplementedError('invalid dataset: {}'.format(config.simclr.train.dataset))

        return train_dataset, val_dataset, test_dataset, classes

    @staticmethod
    def get_simclr_loader(config, train_dataset):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
            sampler=None,
        )

        return train_loader

    @staticmethod
    def get_loaders(config, train_dataset, val_dataset, test_dataset):

        # thoose datasets have no separate val set so the val images have to be samples from the same dataset
        if config.simclr.train.dataset == 'CIFAR10' or config.simclr.train.dataset == 'STL10':

            valid_size = 0.2
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(valid_size * num_train))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.simclr.train.batch_size, sampler=train_sampler,
                num_workers=config.simclr.train.num_workers,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config.simclr.train.batch_size, sampler=valid_sampler,
                num_workers=config.simclr.train.num_workers,
            )
        else:

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.simclr.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=config.simclr.train.num_workers,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.simclr.train.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=config.simclr.train.num_workers,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
        )

        return train_loader, val_loader, test_loader
