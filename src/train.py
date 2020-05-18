from src.fns import model_fn, eval_input_fn, train_input_fn
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
import shutil
import os


def serving_input_receiver_fn():

    inputs = {"x": tf.placeholder(shape=[28, 28], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train(model_dir,
          batch_size=32,
          epochs=10,
          checkpoint_path=None,
          reset=False):

    if os.path.exists(checkpoint_path) is False:
        raise Exception("Checkpoint doesn't exist")

    clear = True
    if clear is True:
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    nb_samples = train_x.shape[0]
    steps_per_epoch = nb_samples // batch_size
    train_steps = steps_per_epoch * epochs
    eval_steps = min(50, test_x.shape[0] // batch_size)

    training_config = tf.estimator.RunConfig(model_dir=model_dir,
                                             save_summary_steps=10,
                                             keep_checkpoint_max=5,
                                             save_checkpoints_steps=steps_per_epoch / 10,
                                             log_step_count_steps=steps_per_epoch / 10)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=training_config,
                                       warm_start_from=checkpoint_path)

    train_spec = tf.estimator.TrainSpec(lambda: train_input_fn(train_x, train_y, batch_size, epochs),
                                        max_steps=train_steps)

    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(test_x, test_y, batch_size),
                                      steps=eval_steps,
                                      throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    estimator.export_savedmodel(export_dir_base=model_dir,
                                serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == "__main__":

    file_path = os.path.abspath(__file__)
    project_dir = Path(file_path).resolve().parents[1]
    model_dir = "{}/model/v1".format(project_dir)

    checkpoint_name = "model.ckpt-188"
    checkpoint_path = "{}/{}".format(model_dir, checkpoint_name)

    batch_size = 32
    epochs = 10

    train(model_dir=model_dir,
          batch_size=batch_size,
          epochs=epochs)
