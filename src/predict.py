from src.fns import model_fn, eval_input_fn
import tensorflow as tf
import click
import imageio


@click.command()
@click.option('--checkpoint')
@click.option('--image_path')
def predict(checkpoint,
            image_path):

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       warm_start_from=checkpoint)

    image_arr = imageio.imread(image_path).reshape(1, 28, 28, -1)

    predictions = estimator.predict(input_fn=lambda: eval_input_fn(image_arr,
                                                                   labels=None,
                                                                   batch_size=32))

    for prediction in predictions:
        print(prediction)


if __name__ == "__main__":
    predict()
