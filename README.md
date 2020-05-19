# Fashion MNIST with Tensorflow 1.15

## Set-up environment

Run the following command:
```make venv```

It will create a python environment and install all the packages specified in requirements.txt.<br/>
It will also install the current package.<br/>

Make sure to activate the virtual environment with the following command:<br/>
```source activate ezmile_venv/bin/activate```

## Fashion MNIST

You can find a quick EDA for Fashion MNIST dataset in ```notebooks/EDA.ipynb```.

## Train model locally

You can train the model using the following command line:<br/>
```python -m src.train```<br/>

You can provide the following optional arguments:<br/>
 ```--model_dir``` Directory where to store model checkpoints.<br/>
 ```--batch_size``` Batch size to use for training.<br/>
 ```--epochs``` Number of epochs.<br/>
 ```--checkpoint``` Filepath to checkpoint to use as starting point for training.<br/>
 ```--reset``` Clear model dir and restart training from scratch.<br/>

Check training and evaluation accuracy with Tensorboard:<br/>
```myvenv/bin/tensorboard --logdir /path/to/dir```<br/>

## Train model on Colab

If your machine doesn't have a GPU, you may find advantage in using google colab.<br/>
Just upload notebook ```notebooks/train.ipynb``` onto colab and then run it there with GPU support.

## Make predictions

You can generate random samples to use for predictions:<br/>
```python -m src.generate_samples```

You can then run predictions using the following command:<br/>
```python -m src.predict --checkpoint path/to/checkpoint --image_path /path/to/image```

Example:<br/>
```python -m src.predict --checkpoint model/v1/model.ckpt-7332 --image_path samples/1.png```
