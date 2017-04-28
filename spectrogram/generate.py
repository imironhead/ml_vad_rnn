"""
"""
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import os
import random
import scipy.io.wavfile as wav
import scipy.misc
import tensorflow as tf

from six.moves import range


tf.app.flags.DEFINE_string('voice-dir-path', None, '')
tf.app.flags.DEFINE_string('music-dir-path', None, '')
tf.app.flags.DEFINE_string('train-dir-path', None, '')
tf.app.flags.DEFINE_integer('size', 10, '')

FLAGS = tf.app.flags.FLAGS


class WavSamples(object):
    """
    """
    def __init__(self, path):
        """
        """
        self._paths = [os.path.join(path, n)
                       for n in os.listdir(path) if n.endswith('.wav')]

        self._indice = np.random.permutation(len(self._paths))

        self._indice_position = 0

        self._samples = []

    def next_batch(self, size):
        """
        """
        while len(self._samples) < size:
            if self._indice_position + 1 == len(self._indice):
                np.random.shuffle(self._indice)

            self._indice_position = \
                (self._indice_position + 1) % len(self._indice)

            _, samples = wav.read(
                self._paths[self._indice[self._indice_position]])

            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 65535.0

            scale = max(np.abs(samples))

            samples = samples / scale

            self._samples = np.concatenate([self._samples, samples])

        result, self._samples = self._samples[:size], self._samples[size:]

        return result


def short_time_fourier_transform(
        signal, window_size, step_size, window_weights_fn=np.hanning):
    """
    """
    signal_size = signal.shape[0]

    timebin_size = 1 + (signal_size - window_size) / step_size

    element_stride = signal.strides[-1]

    results = stride_tricks.as_strided(
        signal,
        shape=(timebin_size, window_size),
        strides=(element_stride * step_size, element_stride))

    results = results * window_weights_fn(window_size)

    return np.fft.rfft(results)


def logscale_spectrogram(spectrum, factor=1.0):
    """
    """
    timebins, freqbins = np.shape(spectrum)

    scale = np.linspace(0.0, 1.0, freqbins) ** factor
    scale = scale * (freqbins - 1) / np.max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)

    spectrogram = np.complex128(np.zeros((timebins, len(scale))))

    scale = np.append(scale, freqbins)

    for i in range(len(scale) - 1):
        spectrogram[:, i] = np.sum(spectrum[:, scale[i]:scale[i+1]], axis=1)

    return spectrogram


def vgg_spectrogram(samples, window_size=446, step_size=40):
    """
    """
    spectrum = short_time_fourier_transform(samples, window_size, step_size)

    spectrogram = logscale_spectrogram(spectrum)

    spectrogram = (np.abs(spectrogram) + 10e-12) / 10e-6

    spectrogram = 20.0 * np.log10(spectrogram)

    return spectrogram


def luma_to_heatmap(luma):
    """
    hsl
    """
    row, col = luma.shape

    rgb = np.zeros((row, col, 3))

    lo = 0.0
    hi = 1.0

    luma = 4.0 * (1.0 - luma / 255.0)

    h = np.floor(luma)
    k = luma - h
    h = h.astype(np.int32)

    loc = (h == 0)
    rgb[loc, 0] = hi
    rgb[loc, 1] = lo + k[loc] * (hi - lo)
    rgb[loc, 2] = lo

    loc = (h == 1)
    rgb[loc, 0] = hi + k[loc] * (lo - hi)
    rgb[loc, 1] = hi
    rgb[loc, 2] = lo

    loc = (h == 2)
    rgb[loc, 0] = lo
    rgb[loc, 1] = hi
    rgb[loc, 2] = lo + k[loc] * (hi - lo)

    loc = (h == 3)
    rgb[loc, 0] = lo
    rgb[loc, 1] = hi + k[loc] * (lo - hi)
    rgb[loc, 2] = hi

    return rgb * 255.0


def generate_vgg_spectrogram(
        voice_samples, music_samples, width, height, step_size):
    """
    """
    voice_batch = voice_samples.next_batch(width)
    music_batch = music_samples.next_batch(width)
    noise_batch = np.random.uniform(size=(width)) * 0.01

    alpha = np.random.rand() * 0.3 + 0.7

    voice_batch = voice_batch * alpha
    music_batch = music_batch * (1.0 - alpha) + noise_batch

    positive_batch = music_batch + voice_batch
    negative_batch = music_batch

    positive_spectrogram = vgg_spectrogram(positive_batch, 446, step_size)
    negative_spectrogram = vgg_spectrogram(negative_batch, 446, step_size)

    return positive_spectrogram.T, negative_spectrogram.T


def main(_):
    """
    """
    voice_samples = WavSamples(FLAGS.voice_dir_path)
    music_samples = WavSamples(FLAGS.music_dir_path)

    positive_spectrogram, negative_spectrogram = \
        generate_vgg_spectrogram(
            voice_samples, music_samples, 2230 * 50, 224, 100)

    for i in range(FLAGS.size):
        if i % 1000 == 0:
            print('{} / {}'.format(i, FLAGS.size))

        if positive_spectrogram.shape[1] < 1000:
            new_positive_spectrogram, new_negative_spectrogram = \
                generate_vgg_spectrogram(
                    voice_samples, music_samples, 2230 * 50, 224, 100)

            positive_spectrogram = np.concatenate(
                [positive_spectrogram, new_positive_spectrogram], axis=1)

            negative_spectrogram = np.concatenate(
                [negative_spectrogram, new_negative_spectrogram], axis=1)

        name = '{:032x}.png'.format(random.getrandbits(128))

        path_positive = os.path.join(FLAGS.train_dir_path, '1_' + name)
        path_negative = os.path.join(FLAGS.train_dir_path, '0_' + name)

        scipy.misc.imsave(path_positive, positive_spectrogram[:, :224])
        scipy.misc.imsave(path_negative, negative_spectrogram[:, :224])

        positive_spectrogram = positive_spectrogram[:, 224:]
        negative_spectrogram = negative_spectrogram[:, 224:]


if __name__ == '__main__':
    tf.app.run()
