"""
"""
import os
import tensorflow as tf

from classifier import Classifier
from spectrogram_streams import wav_stream, mixer_spectrogram_stream

tf.app.flags.DEFINE_string('ckpt-dir-path', './classifier/ckpts/', '')
tf.app.flags.DEFINE_string('logs-dir-path', './classifier/logs/', '')
tf.app.flags.DEFINE_string('bgs-data-path', None, '')
tf.app.flags.DEFINE_string('fgs-data-path', None, '')
tf.app.flags.DEFINE_integer('batch-size', 64, '')
tf.app.flags.DEFINE_integer('bin-step-size', 100, '')
tf.app.flags.DEFINE_integer('img-step-bins', 128, '')
tf.app.flags.DEFINE_integer('spectrogram-size', 128, '')

FLAGS = tf.app.flags.FLAGS


def build_summaries(model):
    """
    """
    summary_metrics = tf.summary.merge([
        tf.summary.scalar('metrics/loss', model.loss),
        tf.summary.scalar('metrics/accuracy', model.metrics_accuracy),
    ])

    return {
        'summary': summary_metrics,
    }


def train():
    """
    """
    bgs_generator = wav_stream(FLAGS.bgs_data_path, 1024)
    fgs_generator = wav_stream(FLAGS.fgs_data_path, 1024)

    data_stream = mixer_spectrogram_stream(
        bgs_generator, fgs_generator, batch_size=FLAGS.batch_size,
        frequency_size=FLAGS.spectrogram_size,
        bin_step_size=FLAGS.bin_step_size, img_step_bins=FLAGS.img_step_bins)

    checkpoint_source_path = tf.train.latest_checkpoint(
        FLAGS.ckpt_dir_path)
    checkpoint_target_path = os.path.join(
        FLAGS.ckpt_dir_path, 'model.ckpt')

    model = Classifier(FLAGS.spectrogram_size)

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if checkpoint_source_path is not None:
            tf.train.Saver().restore(session, checkpoint_source_path)

        # give up overlapped old data
        global_step = session.run(model.global_step)

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=global_step)

        while True:
            batch = next(data_stream)

            fetch = {
                'step': model.global_step,
                'loss': model.loss,
                'trainer': model.trainer,
                'summary': summaries['summary'],
            }

            feeds = {
                model.images: batch['images'],
                model.labels: batch['labels'],
            }

            fetched = session.run(fetch, feeds)

            reporter.add_summary(fetched['summary'], fetched['step'])

            print('[{}]: {}'.format(fetched['step'], fetched['loss']))

            if fetched['step'] % 1000 == 0:
                tf.train.Saver().save(
                    session,
                    checkpoint_target_path,
                    global_step=model.global_step)


def main(_):
    """
    """
    train()


if __name__ == '__main__':
    tf.app.run()
