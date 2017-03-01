"""
"""
import numpy as np
import os
import scipy.io.wavfile as wav
import sys

from python_speech_features import mfcc
from util import enum_wav_files, mix_sound_path


def mfcc_info(sample_rate, window_size, window_step, num_cep):
    """
    """
    name_window_size = int(window_size * sample_rate)
    name_window_step = int(window_step * sample_rate)

    path_home = os.path.expanduser('~')

    path_npz = 'data/vad/{}/mean_std_{}_{}_{}.npz'.format(
        sample_rate, name_window_size, name_window_step, num_cep)
    path_npz = os.path.join(path_home, path_npz)

    if os.path.isfile(path_npz):
        print 'exist!!  {}'.format(path_npz)
        return

    print 'collecting mfcc info'

    source_dir = mix_sound_path(sample_rate)
    wav_file_names = enum_wav_files(source_dir)
    path_wavs = [os.path.join(source_dir, name) for name in wav_file_names]

    mean = np.zeros((num_cep))
    num_features = 0

    # mean
    for path_wav in path_wavs:
        print 'computing mean [ {} ]'.format(num_features)

        _, samples = wav.read(path_wav)

        if len(samples.shape) > 1:
            samples = samples[:, 0]

        features = mfcc(
            samples,
            sample_rate,
            winlen=window_size,
            winstep=window_step,
            numcep=num_cep)

        num_features += features.shape[0]
        mean += np.sum(features, axis=0)

    mean /= num_features

    std = np.zeros((num_cep))

    std_test = None

    # std
    for idx, path_wav in enumerate(path_wavs):
        print 'computing std [ {} ]'.format(idx)

        _, samples = wav.read(path_wav)

        if len(samples.shape) > 1:
            samples = samples[:, 0]

        features = mfcc(
            samples,
            sample_rate,
            winlen=window_size,
            winstep=window_step,
            numcep=num_cep)

        std += np.sum((features - mean) ** 2, axis=0)

        if std_test is None:
            std_test = features
        else:
            std_test = np.vstack((std_test, features))

    std = np.sqrt(std / num_features)

    std_test = np.std(std_test, axis=0)

    print 'mean: {}'.format(mean)
    print 'std: {}'.format(std)
    print 'test std: {}'.format(std_test)
    print 'diff std: {}'.format(np.sum((std - std_test) ** 2.0))

    # dump
    np.savez(path_npz, mean=mean, std=std)


def main():
    """
    """
    if len(sys.argv) != 5:
        print 'python norm_mfcc sample_rate window_size window_step num_cep'
        print 'sample_rate: wav sample rate.'
        print 'window_size: window size in seconds'
        print 'window_step: window step in seconds.'
        print 'num_cep: cepstrum size.'
        return

    sample_rate = int(sys.argv[1])
    window_size = float(sys.argv[2])
    window_step = float(sys.argv[3])
    num_cepstrum = int(sys.argv[4])

    mfcc_info(sample_rate, window_size, window_step, num_cepstrum)


if __name__ == '__main__':
    main()
