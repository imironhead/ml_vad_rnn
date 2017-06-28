"""
"""
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import os
import scipy.io.wavfile as wav
import tensorflow as tf

from spectrogram import generate_spectrogram


tf.app.flags.DEFINE_string('source-dir-path', None, '')
tf.app.flags.DEFINE_string('target-dir-path', None, '')

FLAGS = tf.app.flags.FLAGS


def wav_stream(path, size, infinite=True):
    """
    generator
    """
    # collect all wavs
    if os.path.isdir(path):
        wavs = [n for n in os.listdir(path) if n.endswith('.wav')]

        paths = [os.path.join(path, w) for w in wavs]
    elif os.path.isfile(path) and path.endswith('.wav'):
        paths = [path]
    else:
        raise Exception('invalid path: {}'.format(path))

    # random indice to load wavs
    indice = np.random.permutation(len(paths))

    # indice to complete an epoch
    indice_position = 0

    # buffer to queue wav samples
    samples = []

    while True:
        # pull new samples from next wav files.
        while len(samples) < size:
            _, new_samples = wav.read(paths[indice[indice_position]])

            if new_samples.dtype == np.int16:
                new_samples = new_samples.astype(np.float32) / 65535.0

            new_samples /= max(np.abs(new_samples))

            samples = np.concatenate([samples, new_samples])

            # book keeping
            indice_position = (indice_position + 1) % len(indice)

            if indice_position == 0:
                if infinite:
                    np.random.shuffle(indice)
                else:
                    # out of samples
                    return

        output, samples = samples[:size], samples[size:]

        yield output


def spectrogram_stream(
        wav_generator, batch_size=64, frequency_size=224, bin_step_size=100,
        img_step_bins=224):
    """
    audio for prediction, no manipulation, empty labels
    """
    # assume frequency_size is even
    window_size = frequency_size * 2 - 2

    # sample queues
    wav_samples = []

    # size lower bound to generate next batch
    num_bins = img_step_bins * (batch_size - 1) + frequency_size
    reserved = (num_bins - 1) * bin_step_size + window_size

    # size to drop from sample queues after each batch (there are used)
    batch_step_size = batch_size * img_step_bins * bin_step_size

    while True:
        # pull samples from sources to fill sample queues.
        while len(wav_samples) < reserved:
            new_wav_samples = next(wav_generator)

            wav_samples = np.concatenate([wav_samples, new_wav_samples])

        # generate new spectrograms
        wav_spectrogram = generate_spectrogram(
            wav_samples[:reserved], window_size, bin_step_size)

        # drop used samples
        wav_samples = wav_samples[batch_step_size:]

        shape = (batch_size, frequency_size, frequency_size)

        strides = (
            wav_spectrogram.strides[1] * img_step_bins,
            wav_spectrogram.strides[0],
            wav_spectrogram.strides[1],
        )

        # reshape large/single spectrograms to batches.
        wav_spectrogram = stride_tricks.as_strided(
            wav_spectrogram, shape=shape, strides=strides)

        wav_spectrogram = np.vstack(wav_spectrogram)

        wav_spectrogram = np.reshape(
            wav_spectrogram, (batch_size, frequency_size, frequency_size, 1))

        yield {
            'images': wav_spectrogram,
        }


def mixer_spectrogram_stream(
        bgs_generator, fgs_generator, batch_size=64, frequency_size=224,
        bin_step_size=100, img_step_bins=224):
    """
    assume bgs_generator and fgs_generator return the same size of samples in
    each step.
    """
    # assume batch_size is even
    batch_size /= 2

    # assume frequency_size is even
    window_size = frequency_size * 2 - 2

    # sample queues
    bgs_samples = []
    mix_samples = []

    # size lower bound to generate next batch
    num_bins = img_step_bins * (batch_size - 1) + frequency_size
    reserved = (num_bins - 1) * bin_step_size + window_size

    # size to drop from sample queues after each batch (there are used)
    batch_step_size = batch_size * img_step_bins * bin_step_size

    # always yield batch_size spectrograms which have no voices,
    # folloed by batch_size spectrograms which have voices.
    labels = np.concatenate([
        np.zeros((batch_size), dtype=np.int32),
        np.ones((batch_size), dtype=np.int32)])

    while True:
        # pull samples from sources to fill sample queues.
        while len(bgs_samples) < reserved:
            new_bgs_samples = next(bgs_generator)
            new_fgs_samples = next(fgs_generator)
            new_rdm_samples = np.random.uniform(
                size=(len(new_bgs_samples))) * 0.01

            alpha = np.random.rand() * 0.3 + 0.7

            new_fgs_samples = new_fgs_samples * alpha
            new_bgs_samples = new_bgs_samples * (1.0 - alpha) + new_rdm_samples
            new_mix_samples = new_fgs_samples + new_bgs_samples

            bgs_samples = np.concatenate([bgs_samples, new_bgs_samples])
            mix_samples = np.concatenate([mix_samples, new_mix_samples])

        # generate new spectrograms
        bgs_spectrogram = generate_spectrogram(
            bgs_samples[:reserved], window_size, bin_step_size)
        mix_spectrogram = generate_spectrogram(
            mix_samples[:reserved], window_size, bin_step_size)

        # drop used samples
        bgs_samples = bgs_samples[batch_step_size:]
        mix_samples = mix_samples[batch_step_size:]

        shape = (batch_size, frequency_size, frequency_size)

        strides = (
            bgs_spectrogram.strides[1] * img_step_bins,
            bgs_spectrogram.strides[0],
            bgs_spectrogram.strides[1],
        )

        # reshape large/single spectrograms to batches.
        bgs_spectrogram = stride_tricks.as_strided(
            bgs_spectrogram, shape=shape, strides=strides)
        mix_spectrogram = stride_tricks.as_strided(
            mix_spectrogram, shape=shape, strides=strides)

        bgs_spectrogram = np.vstack(bgs_spectrogram)
        mix_spectrogram = np.vstack(mix_spectrogram)

        bgs_spectrogram = np.reshape(
            bgs_spectrogram, (batch_size, frequency_size, frequency_size, 1))
        mix_spectrogram = np.reshape(
            mix_spectrogram, (batch_size, frequency_size, frequency_size, 1))

        yield {
            'labels': labels,
            'images': np.concatenate([bgs_spectrogram, mix_spectrogram]),
        }


def cued_spectrogram_stream():
    """
    audio + subtitle, try to keep half true, half false
    """
    pass


if __name__ == '__main__':
    # simple test
    import scipy.misc

    for frequency_size in [128, 256]:
        for bin_step_size in [32, 64, 128]:
            wav_generator = wav_stream(
                FLAGS.source_dir_path, 1024, infinite=False)

            stream = spectrogram_stream(
                wav_generator,
                batch_size=1,
                frequency_size=frequency_size,
                bin_step_size=bin_step_size,
                img_step_bins=frequency_size)

            for i, batch in enumerate(stream):
                for j, image in enumerate(batch['images']):
                    new_path = os.path.join(
                        FLAGS.target_dir_path, '{}_{}_{:04}_{:04}.png'.format(
                            frequency_size, bin_step_size, i, j))

                    image = np.reshape(image, [frequency_size] * 2)

                    scipy.misc.imsave(new_path, image)

                    print(new_path)
