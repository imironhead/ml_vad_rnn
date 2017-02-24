"""
"""
import numpy as np
import os
import scipy.io.wavfile as wav
import sys

from util import background_raw_sound_path, foreground_raw_sound_path
from util import enum_wav_files, mix_sound_path


def duration_in_samples(path_to_wav, target_sample_rate):
    """
    get duration of a wav file (in samples)
    """
    if not os.path.isfile(path_to_wav):
        return 0

    sample_rate, samples = wav.read(path_to_wav)

    if sample_rate != target_sample_rate:
        raise Exception('found wav with invalid sample rate')

    return len(samples)


def report_fgs(sample_rate):
    """
    """
    print 'collecting foreground sound info (speech)'

    source_dir = foreground_raw_sound_path(sample_rate)

    num_samples = 0

    for f in enum_wav_files(source_dir):
        source_wav = os.path.join(source_dir, f)

        num_samples += duration_in_samples(source_wav, sample_rate)

    s = num_samples / sample_rate

    m, s = s / 60, s % 60
    h, m = m / 60, m % 60

    print 'fgs totoal length for {} Hz: {}:{:02}:{:02}'.format(
        sample_rate, h, m, s)


def report_bgs(sample_rate):
    """
    """
    print 'collecting background sound info (noise)'

    source_dir = background_raw_sound_path(sample_rate)

    num_samples = 0

    for f in enum_wav_files(source_dir):
        source_wav = os.path.join(source_dir, f)

        num_samples += duration_in_samples(source_wav, sample_rate)

    s = num_samples / sample_rate

    m, s = s / 60, s % 60
    h, m = m / 60, m % 60

    print 'bgs totoal length for {} Hz: {}:{:02}:{:02}'.format(
        sample_rate, h, m, s)


def report_mfcc(sample_rate):
    """
    """
    no_npz = True

    source_path = mix_sound_path(sample_rate)

    for name in os.listdir(source_path):
        if not name.endswith('.npz'):
            continue

        if not name.startswith('mean_std_'):
            continue

        no_npz = False

        segments = name.replace('.', '_').split('_')

        window_size = float(segments[2]) / sample_rate
        window_step = float(segments[3]) / sample_rate
        num_cep = int(segments[4])

        std_mean = np.load(os.path.join(source_path, name))

        print 'mean for {}/{}/{}:'.format(window_size, window_step, num_cep)
        print std_mean['mean']

        print 'std for {}/{}/{}:'.format(window_size, window_step, num_cep)
        print std_mean['std']

    if no_npz:
        print 'found no pre-calculated mean and std!'


def main():
    """
    """
    if len(sys.argv) != 2:
        raise Exception('need sample rate')

    sample_rate = int(sys.argv[1])

    report_fgs(sample_rate)
    report_bgs(sample_rate)
    report_mfcc(sample_rate)


if __name__ == '__main__':
    main()
