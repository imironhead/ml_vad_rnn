"""
rename all raw audio files with md5(file_contents) name.
"""
import os
import re

from util import background_raw_sound_path, foreground_raw_sound_path
from util import md5


def is_renamed(name):
    """
    check if the file name is a md5 string.
    """
    return len(name) >= 33 and \
        name[32] == '.' and \
        re.match(r"[a-f,0-9]{32}", name[:32])


def main():
    """
    rename raw audio files.
    """
    dirs = [
        background_raw_sound_path(), foreground_raw_sound_path()]

    for d in dirs:
        for old_name in os.listdir(d):
            if is_renamed(old_name):
                continue

            name, ext = os.path.splitext(old_name)

            old_name = os.path.join(d, old_name)
            new_name = md5(old_name)
            new_name = os.path.join(d, new_name + '.' + ext)

            if os.path.isfile(new_name):
                continue

            os.system('mv {} {}'.format(old_name, new_name))

            print new_name


if __name__ == '__main__':
    main()
