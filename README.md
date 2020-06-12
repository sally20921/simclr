# What is contrastive learning?
- a paradigm of self-supervised learning
- an approach to formulate the task of finding similar and dissimilar things for an ML model
- one can train a machine learning model to classify between similar and dissimilar images
- inner working of contrastive learning can be formulated as a score function, which  is a metric
that measure the similarity between two features. 
- x+ is data point similar to x, referred to as positive sample
- x- is data point dissimilar to x, referred to as negative sample
- a softmax classifier can be built that classifies positive and negative samples correctly


# train the model 

```
sh setup.sh
conda activate simclr
pip3 install -r requirements.txt
```

```
python3 main.py
```

# test the model
```
python3 -m testing.logistic_regression with model_path=./logs/0 
```

# result
- SimCLR+Linear eval.
- batch size: 256
- ResNet: ResNet50
- epochs: 100
- optimizer: Adam
- dataset: STL-10
- result:  0.829
