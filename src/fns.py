import tensorflow as tf
from src.model import get_model
from src.transformations import reshape_and_cast_inputs, \
                                preprocess_image


def train_input_fn(features, labels, batch_size, epochs, shuffle_size=1000):

    dataset = tf.data.Dataset.from_tensor_slices(({"x": features}, labels))
    dataset = dataset.map(reshape_and_cast_inputs)
    dataset = dataset.shuffle(shuffle_size).repeat(epochs).batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size, shuffle_size=1000):

    if labels is None:
        inputs = {"x": features}
    else:
        inputs = ({"x": features}, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if labels is None:
        dataset = dataset.map(preprocess_image)
    else:
        dataset = dataset.map(reshape_and_cast_inputs)

    dataset = dataset.cache().shuffle(shuffle_size).batch(batch_size)

    return dataset


def model_fn(features, labels, mode, params):

    features = tf.reshape(features["x"], [-1, 28, 28, 1])
    model = get_model()

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
    elif mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features, training=False)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(features, training=True)
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
