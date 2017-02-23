"""
"""
import hashlib
import os


def make_dir(path):
    """
    make a directory if it is not exist.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def md5(fname):
    hash_md5 = hashlib.md5()

    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def enum_audio_files(source_dir):
    """
    enum all mp3 and wav files in a specific directory.
    """
    return [n for n in os.listdir(source_dir)
            if n.endswith('.mp3') or n.endswith('.wav')]


def enum_wav_files(source_dir):
    """
    enum all wav files in a specific directory.
    """
    return [n for n in os.listdir(source_dir) if n.endswith('.wav')]


def raw_sound_path(fg_or_bg='raw_fgs', sample_rate=None):
    """
    return path to the directory that comtains the raw sound with specific
    sampling rate.
    """
    sample_rate = '' if sample_rate is None else '_{}'.format(sample_rate)

    return os.path.join(
        os.path.expanduser('~'),
        'data/vad/{}{}/'.format(fg_or_bg, sample_rate))


def background_raw_sound_path(sample_rate=None):
    """
    return path to the directory that contains raw background sound files.
    """
    return raw_sound_path('raw_bgs', sample_rate)


def foreground_raw_sound_path(sample_rate=None):
    """
    return path to the directory that contains raw foreground sound files.
    """
    return raw_sound_path('raw_fgs', sample_rate)


def mix_sound_path(sample_rate):
    """
    """
    return os.path.join(
        os.path.expanduser('~'),
        'data/vad/{}/'.format(sample_rate))
