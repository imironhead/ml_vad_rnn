"""
"""
import os
import tensorflow as tf

from classifier import Classifier
from spectrogram_streams import wav_stream, spectrogram_stream

tf.app.flags.DEFINE_string('ckpt-dir-path', './classifier/ckpts/', '')
tf.app.flags.DEFINE_string('wav-data-path', None, '')
tf.app.flags.DEFINE_string('srt-result-path', None, '')
tf.app.flags.DEFINE_integer('batch-size', 64, '')
tf.app.flags.DEFINE_integer('bin-step-size', 100, '')
tf.app.flags.DEFINE_integer('img-step-bins', 128, '')
tf.app.flags.DEFINE_integer('spectrogram-size', 128, '')
tf.app.flags.DEFINE_integer('sample-rate', 4000, '')

FLAGS = tf.app.flags.FLAGS


class SrtWriter(object):
    @staticmethod
    def timestamp(idx, step, sample_rate):
        """
        """
        t = idx * step * 1000 / sample_rate

        H, t = t / 3600000, t % 3600000
        M, t = t / 60000, t % 60000
        S, m = t / 1000, t % 1000

        return "{}:{}:{},{}".format(H, M, S, m)

    @staticmethod
    def save(path, features, sample_rate, window_step):
        """
        """
        with open(path, 'w') as srt:
            caption_idx = 0
            feature_idx = 0

            while True:
                while True:
                    if feature_idx >= len(features):
                        break
                    if features[feature_idx] == 1:
                        break
                    feature_idx += 1

                if feature_idx >= len(features):
                    break

                time_head = \
                    SrtWriter.timestamp(feature_idx, window_step, sample_rate)

                while True:
                    if feature_idx >= len(features):
                        break
                    if features[feature_idx] == 0:
                        break
                    feature_idx += 1

                time_tail = \
                    SrtWriter.timestamp(feature_idx, window_step, sample_rate)

                srt.write("{}\n".format(caption_idx))
                srt.write("{} --> {}\n".format(time_head, time_tail))
                srt.write("!@#$%^\n\n")

                caption_idx += 1


def predict():
    """
    """
    wav_generator = wav_stream(FLAGS.wav_data_path, 1024, infinite=False)

    data_stream = spectrogram_stream(
        wav_generator, batch_size=FLAGS.batch_size,
        frequency_size=FLAGS.spectrogram_size,
        bin_step_size=FLAGS.bin_step_size, img_step_bins=FLAGS.img_step_bins)

    checkpoint_source_path = tf.train.latest_checkpoint(
        FLAGS.ckpt_dir_path)

    masks = []

    model = Classifier(FLAGS.spectrogram_size)

    if not os.path.isfile(checkpoint_source_path):
        raise Exception('invalid ckpt: {}'.format(checkpoint_source_path))

    with tf.Session() as session:
        tf.train.Saver().restore(session, checkpoint_source_path)

        for batch in data_stream:
            feeds = {model.images: batch['images']}

            predictions = session.run(model.predictions, feeds)

            masks.extend(predictions)

    SrtWriter.save(
        FLAGS.srt_result_path, masks, FLAGS.sample_rate, FLAGS.img_step_bins)


def main(_):
    """
    """
    predict()


if __name__ == '__main__':
    tf.app.run()
