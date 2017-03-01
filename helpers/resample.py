"""
"""
import os
import sys

from util import make_dir, enum_audio_files
from util import background_raw_sound_path, foreground_raw_sound_path


def main():
    """
    resample raw fg/bg sound files with specific sampling rate.
    """
    if len(sys.argv) != 2:
        raise 'need sample rate'

    sample_rate = sys.argv[1]

    dirs = [
        (background_raw_sound_path(), background_raw_sound_path(sample_rate)),
        (foreground_raw_sound_path(), foreground_raw_sound_path(sample_rate))]

    for source_dir, target_dir in dirs:
        make_dir(target_dir)

        source_files = enum_audio_files(source_dir)

        for f in source_files:
            name, ext = os.path.splitext(f)

            source_file = os.path.join(source_dir, f)
            target_file = os.path.join(target_dir, name + '.wav')

            if os.path.isfile(target_file):
                continue

            command = 'ffmpeg -i {} -acodec pcm_f32le -ac 1 -ar {} {}' \
                .format(source_file, sample_rate, target_file)

            os.system(command)


if __name__ == '__main__':
    main()
