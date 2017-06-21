import glob
import os
import tensorflow as tf


tf.app.flags.DEFINE_string('source-dir-path', None, '')
tf.app.flags.DEFINE_string('target-dir-path', None, '')
tf.app.flags.DEFINE_integer('sample-rate', 4000, '')

FLAGS = tf.app.flags.FLAGS


def main(_):
    """
    """
    print('start to resample in {}'.format(FLAGS.sample_rate))

    source_paths = []

    source_paths.extend(
        glob.glob(os.path.join(FLAGS.source_dir_path, '*.mp3')))
    source_paths.extend(
        glob.glob(os.path.join(FLAGS.source_dir_path, '*.wav')))

    for idx, path in enumerate(source_paths):
        _, name = os.path.split(path)
        name, ext = os.path.splitext(name)

        target_wav_path = os.path.join(FLAGS.target_dir_path, name + '.wav')

        command = 'ffmpeg'
        command += ' -i {}'.format(path)
        command += ' -acodec pcm_f32le'
        command += ' -ac 1'
        command += ' -ar {}'.format(FLAGS.sample_rate)
        command += ' -map_metadata -1 -write_xing 0'
        command += ' {}'.format(target_wav_path)

        os.system(command)

        if idx % 100 == 0:
            print('{} / {}'.format(idx, len(source_paths)))


if __name__ == '__main__':
    tf.app.run()
