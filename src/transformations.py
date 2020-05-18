import tensorflow as tf


def preprocess_image(image):

    image["x"] = tf.cast(image["x"], tf.float32)
    image["x"] /= 255

    return image


def reshape_and_cast_inputs(image, label):

    image = preprocess_image(image)
    label = tf.cast(label, tf.int32)

    return image, label
