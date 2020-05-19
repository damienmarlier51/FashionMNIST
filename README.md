# Fashion MNIST with Tensorflow 1.15

## Set-up environment

Run the following command:
```make venv```

It will create a python environment and install all the packages specified in requirements.txt. 
It will also install the current package.

Make sure to activate the virtual environment with the following command:
```source activate ezmile_venv/bin/activate```

## Fashion MNIST

You can find a quick EDA for Fashion MNIST dataset in ```notebooks/EDA.ipynb```.

## Train model locally

You can train the model using the following command line:
```python -m src.train```

You can provide the following optional arguments:
 ```--model_dir``` Directory where to store model checkpoints.
 ```--batch_size``` Batch size to use for training.
 ```--epochs``` Number of epochs.
 ```--checkpoint``` Filepath to checkpoint to use as starting point for training.
 ```--reset``` Clear model dir and restart training from scratch.

Check training and evaluation accuracy with Tensorboard:
```myvenv/bin/tensorboard --logdir /path/to/dir```

## Train model on Colab

If your machine doesn't have a GPU, you may find advantage in using google colab.
Just upload notebook ```notebooks/train.ipynb``` onto colab and then run it there with GPU support.

## Make predictions

You can generate random samples to use for predictions:
```python -m src.generate_samples```

You can then run predictions using the following command:
```python -m src.predict --checkpoint path/to/checkpoint --image_path /path/to/image```

Example:
```python -m src.predict --checkpoint model/v1/model.ckpt-7332 --image_path samples/1.png```
