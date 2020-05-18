# Fashion MNIST with Tensorflow 1.15

## Set-up environment

Run the following command:
```make venv```

It will create a python environment with all the packages specified in requirements.txt installed. It will also install the current package.

Make sure to activate the virtual environment with the following command:
```source activate ezmile_venv/bin/activate```

## Fashion MNIST

You can find a quick EDA for Fashion MNIST dataset in ```notebooks/EDA.ipynb```

## Train model
```

Check training and evaluation accuracy with Tensorboard:
```myvenv/bin/tensorboard --logdir model/v1```

## Make predictions

You can generate random samples to use for predictions:
```python -m src.generate_samples```


```python -m src.predict --checkpoint model/v1/model.ckpt-7332 --image_path samples/1.png```
