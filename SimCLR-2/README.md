# SimCLR

Unofficial PyTorch implementation of _SimCLR: A Simple Framework for Contrastive Learning of Visual Representations_ by Ting Chen et al.

[Link to paper](https://arxiv.org/pdf/2002.05709.pdf)

[Link to official tensorflow based implementation](https://github.com/google-research/simclr)

## Setup

```
source venv/bin/activate
pip install -r requirements.txt 
```

## Config file

This configuration can be used to evaluate the simclr based training of a ResNet model. The trained model can then be 
used to perform a linear evaluation, fine tuning and be converted to onnx. 

```yaml
simclr:

  train:
    batch_size: 512
    num_workers: 8
    start_epoch: 0
    epochs: 100
    # adjust this path to point to the directory where the dataset is located
    data_dir_path: "/home/username/Data"
    dataset: "CIFAR10"
    save_num_epochs: 1
    img_size: 32
    optimizer: "Adam"
    weight_decay: 1.0e-6
    temperature: 0.1

  model:
    resnet: "resnet18"
    normalize: True
    projection_dim: 64

logistic_regression:
  epochs: 200
  learning_rate: 0.001
  batch_size: 512
  momentum: 0.9
  img_size: 32

fine_tuning:
  learning_rate: 0.001
  batch_size: 512
  momentum: 0.9
  step_size: 10
  gamma: 0.1
  epochs: 100
  img_size: 32

onnx:
  batch_size: 512
  img_size: 32
```

## Training 

```
python train_simclr.py PATH_TO_CONFIG_FILE
```

## Linear evaluation

```
python python train_logistic_regression.py PATH_TO_CONFIG_FILE output/PATH_TO_GENERATED_TRAINING_OUTPUT EPOCH_NUM
```

## Fine tuning

```
python python train_classification.py PATH_TO_CONFIG_FILE output/PATH_TO_GENERATED_TRAINING_OUTPUT EPOCH_NUM
```

## Convert to ONNX

```
python convert_to_onnx.py PATH_TO_CONFIG_FILE output/PATH_TO_GENERATED_TRAINING_OUTPUT EPOCH_NUM

```

# Results

_Evaluated accuracy for the test set. Evaluation performed after training the model using simclr._

| Dataset     | Architecture | Batch size | Epochs | Linear Evaluation | Fine Tuning |
| ------------| ------------ | -----------| ------ | ----------------- |------------ |
| CIFAR10     | ResNet50     | 512        | 1000   | 0.7957            | 0.7828      |
| STL10       | ResNet50     | 256        | 1000   | 0.7152            | 0.7166      |
| ImageNet    | ResNet50     | 64         | 70     |                   | 0.4710      |
| iNaturalist | ResNet50     | 64         | 100    |                   | 0.3310      |