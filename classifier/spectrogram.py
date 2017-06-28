"""
"""
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import scipy.io.wavfile as wav
import scipy.misc


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
        spectrogram[:, i] = \
            np.sum(spectrum[:, scale[i]:scale[i+1]], axis=1)

    return spectrogram


def generate_spectrogram(samples, wind_size, step_size):
    """
    """
    spectrogram = short_time_fourier_transform(
        samples, wind_size, step_size)

    # spectrogram = logscale_spectrogram(spectrogram)

    spectrogram = (np.abs(spectrogram) + 10e-12) / 10e-6

    spectrogram = 20.0 * np.log10(spectrogram)

    return spectrogram.T


def batches(samples, batch_size=64, frequency_size=224, bin_step_size=100,
            img_step_bins=224):
    """
    [batch_size, frequency_size, frequency_size, 1]
    """
    # assume frequency_size is even
    window_size = frequency_size * 2 - 2

    position = 0
    num_bins = img_step_bins * (batch_size - 1) + frequency_size
    reserved = (num_bins - 1) * bin_step_size + window_size

    while position + reserved < len(samples):
        spectrogram = generate_spectrogram(
            samples[position:position+reserved],
            window_size,
            bin_step_size)

        position += batch_size * img_step_bins * bin_step_size

        shape = (batch_size, frequency_size, frequency_size)

        strides = (
            spectrogram.strides[1] * img_step_bins,
            spectrogram.strides[0],
            spectrogram.strides[1],
        )

        spectrograms = stride_tricks.as_strided(
            spectrogram, shape=shape, strides=strides)

        spectrograms = np.vstack(spectrograms)

        spectrograms = np.reshape(
            spectrograms, (batch_size, frequency_size, frequency_size, 1))

        yield spectrograms


if __name__ == '__main__':
    """
    for testing
    """
    path_wav = '/Users/lronheadChuang/datasets/movies/'
    path_wav += 'friends/friends_s10e03_8000.wav'

    sample_rate, samples = wav.read(path_wav)

    #
    frequency_size = 128
    bin_step_size = 40
    img_step_bins = 128

    enum_batches = enumerate(batches(
        samples,
        frequency_size=frequency_size,
        bin_step_size=bin_step_size,
        img_step_bins=img_step_bins))

    for i, batch in enum_batches:
        for j, image in enumerate(batch):
            o_path = './test/{:04}_{:04}.png'.format(i, j)

            image += 120.0

            scipy.misc.imsave(
                o_path, np.reshape(image, (frequency_size, frequency_size)))
