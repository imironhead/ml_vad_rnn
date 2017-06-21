"""
"""
import math
import tensorflow as tf

from six.moves import range


class Classifier(object):
    """
    """
    def __init__(self, size):
        """
        """
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        global_step = tf.get_variable(
            name='global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

        images = tf.placeholder(shape=[None, size, size, 1], dtype=tf.float32)

        labels = tf.placeholder(shape=[None], dtype=tf.int32)

        batch_tensors = images

        power = int(math.log(size, 2))

        for layer in range(power - 2):
            batch_tensors = tf.contrib.layers.convolution2d(
                inputs=batch_tensors,
                num_outputs=2 ** (layer + 1),
                kernel_size=3,
                stride=2,
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=tf.contrib.layers.batch_norm,
                weights_initializer=weights_initializer,
                scope='conv_{}'.format(layer))

        batch_tensors = tf.contrib.layers.flatten(batch_tensors)

        #
        batch_tensors = tf.contrib.layers.fully_connected(
            inputs=batch_tensors,
            num_outputs=2,
            weights_initializer=weights_initializer,
            scope='fc')

        loss = tf.contrib.losses.softmax_cross_entropy(
            logits=batch_tensors,
            onehot_labels=tf.one_hot(labels, 2))

        trainer = tf.train.AdamOptimizer(
            learning_rate=0.00001, beta1=0.5, beta2=0.9)

        trainer = trainer.minimize(
            loss,
            global_step=global_step)

        # metrics
        predictions = tf.argmax(batch_tensors, axis=1)
        predictions = tf.cast(predictions, dtype=tf.int32)

        metrics_accuracy = tf.contrib.metrics.accuracy(predictions, labels)

        self._properties = {
            # fetch
            'global_step': global_step,
            'loss': loss,
            'trainer': trainer,
            'metrics_accuracy': metrics_accuracy,
            'predictions': predictions,

            # feed
            'images': images,
            'labels': labels,
        }

    def __getattr__(self, name):
        """
        The properties of this net.
        """
        return self._properties[name]
