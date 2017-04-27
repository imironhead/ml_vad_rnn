"""
"""
import numpy as np
import os
import scipy.misc
import tensorflow as tf

from six.moves import range
from vggnet import VggNet

tf.app.flags.DEFINE_string('train-data-path', None, '')
tf.app.flags.DEFINE_string('vgg19-path', None, '')
tf.app.flags.DEFINE_integer('batch-size', 128, '')

FLAGS = tf.app.flags.FLAGS


class SpectrogramSamples(object):
    """
    """
    def __init__(self, path):
        """
        """
        self._paths = [os.path.join(path, n)
                       for n in os.listdir(path) if n.endswith('.png')]

        self._indice = np.random.permutation(len(self._paths))

        self._indice_position = 0

        self._samples = []

    def next_batch(self, size=64):
        """
        """
        images = np.zeros((size, 224, 224))
        labels = np.ones(size, dtype=np.int32)

        for i in range(size):
            if self._indice_position == len(self._indice):
                self._indice_position = 0

                np.random.shuffle(self._indice)

            image_path = self._paths[self._indice_position]

            is_voice = (image_path[-38] == '1')

            self._indice_position += 1

            images[i] = scipy.misc.imread(image_path)

            labels[i] = 1 if is_voice else 0

        images = np.reshape(images, (size, 224, 224, 1))

        return labels, images


class Classifier(object):
    """
    """
    def __init__(self, vgg19_path):
        """
        """
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        global_step = tf.get_variable(
            name='global_step',
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

        images = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)

        labels = tf.placeholder(shape=[None], dtype=tf.int32)

        batch_tensors = images

        # 1 channel to 3 channels for vgg
        batch_tensors = tf.contrib.layers.convolution2d(
            inputs=batch_tensors,
            num_outputs=3,
            kernel_size=5,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='upstream')

        # connect to vgg
        vgg = VggNet.build_19(vgg19_path, batch_tensors, end_layer='fc7')

        batch_tensors = vgg.downstream

        batch_tensors = tf.contrib.layers.flatten(batch_tensors)

        #
        batch_tensors = tf.contrib.layers.fully_connected(
            inputs=batch_tensors,
            num_outputs=4096,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='mfc7')

        batch_tensors = tf.contrib.layers.fully_connected(
            inputs=batch_tensors,
            num_outputs=2,
            weights_initializer=weights_initializer,
            scope='mfc8')

        loss = tf.contrib.losses.softmax_cross_entropy(
            logits=batch_tensors,
            onehot_labels=tf.one_hot(labels, 2))

        trainer = tf.train.AdamOptimizer(
            learning_rate=0.001, beta1=0.5, beta2=0.9)

        trainer = trainer.minimize(
            loss,
            global_step=global_step)

        self._properties = {
            # fetch
            'global_step': global_step,
            'loss': loss,
            'trainer': trainer,

            # feed
            'images': images,
            'labels': labels,
        }

    def __getattr__(self, name):
        """
        The properties of this net.
        """
        if name in self._properties:
            return self._properties[name]
        else:
            raise Exception('invalid property: {}'.format(name))


def main(_):
    """
    """
    classifier = Classifier(FLAGS.vgg19_path)

    reader = SpectrogramSamples(FLAGS.train_data_path)

    # XLA

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        while True:
            labels, images = reader.next_batch(64)

            fetch = [
                classifier.loss,
                classifier.global_step,
                classifier.trainer,
            ]

            feeds = {
                classifier.images: images,
                classifier.labels: labels,
            }

            loss, step, _ = session.run(fetch, feeds)

            print('[{}]: {}'.format(step, loss))


if __name__ == '__main__':
    tf.app.run()
