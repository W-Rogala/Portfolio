.PHONY: setup data train train-vgg train-vgg-aug train-vgg-small train-vgg-small-aug train-resnet train-resnet-aug evaluate viz-sample viz-aug viz-dist clean

Setup environment
setup: pip install -r requirements.txt pip install -e .

Download and extract data
data: python -m src.data.download

Train models
train-vgg: python -m src.training.train --model vgg --epochs 40 --augment False

train-vgg-aug: python -m src.training.train --model vgg --epochs 150 --augment True

train-vgg-small: python -m src.training.train --model vgg_small --epochs 40 --augment False

train-vgg-small-aug: python -m src.training.train --model vgg_small --epochs 150 --augment True

train-resnet: python -m src.training.train --model resnet --epochs 20 --augment False

train-resnet-aug: python -m src.training.train --model resnet --epochs 100 --augment True

Train all models
train: train-vgg train-vgg-aug train-vgg-small train-vgg-small-aug train-resnet train-resnet-aug

Evaluate a model
evaluate: python -m src.training.evaluate --model_path models/$(MODEL).pth

Visualizations
viz-sample: python -m src.utils.visualization --type sample --samples 8

viz-aug: python -m src.utils.visualization --type augmentation

viz-dist: python -m src.utils.visualization --type distribution

Clean up
clean: rm -rf pycache rm -rf src/pycache rm -rf src/*/pycache rm -rf build rm -rf dist rm -rf *.egg-info

