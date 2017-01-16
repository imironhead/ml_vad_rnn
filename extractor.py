"""
prepare data

expect directory structure for source data:
datasets +-+ cue +-+ raw +-+ test     +-+ *.mp3
                                      +-+ *.srt
                         +-+ training +-+ *.mp3
                                      +-+ *.srt
                         +-+ validate +-+ *.mp3
                                      +-+ *.srt

extract data to:
datasets +-+ cue +-+ sample_rate +-+ test     +-+ *.wav
                                              +-+ *.srt
                                 +-+ training +-+ *.wav
                                              +-+ *.srt
                                 +-+ validate +-+ *.wav
                                              +-+ *.srt

sample_rate can be any integer which is 4000 in the first version.

training data:
    that prefix and first 8 characters of file hash would be served as prefix
    of its associated extracted data. e.g.

    hoc_s03e10_5da5e0bb_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.wav
    hoc_s03e10_5da5e0bb_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.srt

    can be extracted from

    hoc_s03e10_5da5e0bb5e475ef7ac1ee6e59b67678b.mp3
    hoc_s03e10_5da5e0bb5e475ef7ac1ee6e59b67678b.srt

test & validate data:
    data is only transformed. no extraction.

usage: python extractor.py 4000

"""
import hashlib
import os
import sys


class SrtCue(object):
    """
    a subtitle cue object.
    """
    def __init__(self, lines=None, head_ms=None, tail_ms=None, line=None):
        """
        lines: a block in the srt. use lines to initialize the object if lines
               is not None.
        head_ms: the start time of this cue in milliseconds. if lines is None
                 and head_ms is not None
        tail_ms: the end time of this cue in milliseconds. if lines is None and
                 tail_ms is not None
        line: the text of this cue. if lines is None and line is not None.

        format of lines:
        [
            '0',
            '00:00:02,000 --> 00:00:03,445',
            'Hey, what are you doing?'
        ]
        """
        self._head_timestamp_ms = head_ms
        self._tail_timestamp_ms = tail_ms
        self._line = line

        if isinstance(lines, list) and len(lines) >= 3:
            b, e = lines[1].split('-->')

            self._head_timestamp_ms = SrtCue.timestamp_to_ms(b)
            self._tail_timestamp_ms = SrtCue.timestamp_to_ms(e)
            self._line = ' '.join(lines[2:])

    @staticmethod
    def timestamp_to_ms(timestamp):
        """
        convert string 'timestamp' to milliseconds to an integer.

        timestamp: a string in format 'HH:MM:SS,mmm'.
                   (hours:minutes:minutes,milliseconds)
        """
        h, m, s, ms = timestamp.replace(',', ':').split(':')

        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    @staticmethod
    def ms_to_timestamp(ms):
        """
        convert an integer milliseconds to its string form (for srt).

        ms: milliseconds in integer.
        """
        s, ms = ms / 1000, ms % 1000
        m, s = s / 60, s % 60
        h, m = m / 60, m % 60

        return '{:02}:{:02}:{:02},{:03}'.format(h, m, s, ms)

    @staticmethod
    def merge(a, b):
        """
        merge two SrtCue objects into a new SrtCue object.

        a, b: SrtCue objects.
        """
        head_ms = min(a._head_timestamp_ms, b._head_timestamp_ms)
        tail_ms = max(a._tail_timestamp_ms, b._tail_timestamp_ms)
        line = a._line + ' ' + b._line

        return SrtCue(head_ms=head_ms, tail_ms=tail_ms, line=line)

    def line(self):
        """
        return line of this cue.
        """
        return self._line

    def is_valid(self):
        """
        return if this cue is valid (well initialized).
        """
        return self._head_timestamp_ms is not None


class SrtParser(object):
    """
    a class to parse srt file.
    """
    def __init__(self):
        """
        __init__
        """
        self._cues = []

    def cues(self):
        """
        a generator to return cues one by one.
        """
        for cue in self._cues:
            yield cue

    def load(self, path):
        """
        load srt from a file.

        path: path to the srt.
        """
        lines = []

        for line in open(path):
            line = line.rstrip()

            if len(line) == 0 and len(lines) >= 3:
                self._cues.append(SrtCue(lines))

                lines = []
            else:
                lines.append(line)

        # last cue
        if len(lines) > 0:
            self._cues.append(SrtCue(lines))


class SrtWriter(object):
    @staticmethod
    def save(path, cues, offset_ms=0):
        """
        write cues to a srt file.

        path: path to the target file.
        cues: cues we want to export.
        offset_ms: a shift to time spans of all cues. in milliseconds.
        """
        with open(path, 'w') as srt:
            for idx in xrange(len(cues)):
                cue = cues[idx]

                time_head = SrtCue.ms_to_timestamp(
                    cue._head_timestamp_ms - offset_ms)
                time_tail = SrtCue.ms_to_timestamp(
                    cue._tail_timestamp_ms - offset_ms)

                srt.write('{}\n'.format(idx))
                srt.write('{} --> {}\n'.format(time_head, time_tail))
                srt.write(cue.line())
                srt.write('\n\n')


def collect_file_names(path_dir, extension):
    """
    return name list of all files with extension under path_dir.

    path_dir: path to the directory.
    extension: file type for the collection.
    """
    names = []

    for name in os.listdir(path_dir):
        if not os.path.isfile(os.path.join(path_dir, name)):
            continue

        if not name.endswith(extension):
            continue

        names.append(name)

    return names


def ms_to_timestamp(ms):
    """
    convert milliseconds to timestamp for srt.

    ms: milliseconds in integer
    """
    s, ms = ms / 1000, ms % 1000
    m, s = s / 60, s % 60
    h, m = m / 60, m % 60

    return '{:02}:{:02}:{:02}.{:03}'.format(h, m, s, ms)


def make_dir(path):
    """
    create a directory in path if it's not exist.

    path: path to the target diectory.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def prepare_params():
    """
    check source data directories and create target data directories. return
    pathes in a dictionary.
    """
    params = {}

    if len(sys.argv) < 2:
        raise Exception('need sample rate')

    if not sys.argv[1].isdigit():
        raise Exception('need sample rate, must be integer')

    params['sample_rate'] = int(sys.argv[1])

    if params['sample_rate'] < 1:
        raise Exception('need sample rate which is greater than 0')

    path_source_root = './datasets/cue/raw/'
    path_target_root = './datasets/cue/{}/'.format(params['sample_rate'])

    params['path_source_test'] = path_source_root + 'test/'
    params['path_source_training'] = path_source_root + 'training/'
    params['path_source_validate'] = path_source_root + 'validate/'

    params['path_target_test'] = path_target_root + 'test/'
    params['path_target_training'] = path_target_root + 'training/'
    params['path_target_validate'] = path_target_root + 'validate/'

    make_dir(params['path_target_test'])
    make_dir(params['path_target_training'])
    make_dir(params['path_target_validate'])

    return params


def is_valid_line(line):
    """
    return true if the string looks like a cue.

    line: text cue.
    """
    for c in '([<>])/':
        if c in line:
            return False
    return True


def load_cues(path):
    """
    load all cues from srt.

    path: path to the srt.
    """
    srt = SrtParser()

    srt.load(path)

    # clone
    cues = []

    for old_cue in srt.cues():
        new_cue = SrtCue(
            head_ms=old_cue._head_timestamp_ms,
            tail_ms=old_cue._tail_timestamp_ms,
            line=old_cue._line)

        cues.append(new_cue)

    return cues


def extract_span(cues, source_mp3_file, source_hash, target_dir_path,
                 sample_rate, head_ms, tail_ms):
    """
    """
    if cues is None:
        cues = []

    target_hash = '{}{}{}'.format(source_mp3_file, head_ms, tail_ms)
    target_name = hashlib.md5(target_hash).hexdigest()
    target_name = '{}_{}'.format(source_hash, target_name)
    target_srt_path = target_dir_path + target_name + '.srt'
    target_wav_path = target_dir_path + target_name + '.wav'

    if os.path.isfile(target_srt_path):
        return

    SrtWriter.save(target_srt_path, cues, head_ms)

    command = 'ffmpeg'
    command += ' -i {}'.format(source_mp3_file)
    command += ' -acodec pcm_f32le'
    command += ' -ss {}'.format(ms_to_timestamp(head_ms))
    command += ' -t {}'.format(ms_to_timestamp(tail_ms - head_ms))
    command += ' -ac 1'
    command += ' -ar {}'.format(sample_rate)
    command += ' -map_metadata -1 -write_xing 0'
    command += ' {}'.format(target_wav_path)

    os.system(command)


def extract_sound(cues, source_mp3_file, source_hash, target_dir_path,
                  sample_rate):
    """
    """
    previous_ending_ms = 0

    for idx, cue in enumerate(cues):
        span_head_ms = previous_ending_ms
        span_tail_ms = cue._head_timestamp_ms
        previous_ending_ms = cue._tail_timestamp_ms

        while span_head_ms < span_tail_ms:
            # skip span which is less than 5 seconds
            if span_head_ms + 5000 > span_tail_ms:
                break

            # 10 seconds per sample
            head_ms = span_head_ms
            tail_ms = min(head_ms + 10000, span_tail_ms)

            extract_span([], source_mp3_file, source_hash, target_dir_path,
                         sample_rate, head_ms, tail_ms)

            span_head_ms = tail_ms


def extract_voice(cues, source_mp3_file, source_hash, target_dir_path,
                  sample_rate):
    """
    """
    previous_ending_ms = 0

    current_cue = None

    for idx, cue in enumerate(cues):
        if not is_valid_line(cue._line):
            # complete a cue
            if current_cue is not None:
                head_ms = max(
                    current_cue._head_timestamp_ms - 2000,
                    previous_ending_ms)

                tail_ms = min(
                    current_cue._tail_timestamp_ms + 2000,
                    cue._head_timestamp_ms)

                extract_span([current_cue], source_mp3_file, source_hash,
                             target_dir_path, sample_rate, head_ms, tail_ms)

                current_cue = None

            previous_ending_ms = cue._tail_timestamp_ms

            continue

        if current_cue is None:
            current_cue = cue
            continue

        if current_cue._tail_timestamp_ms + 500 > cue._head_timestamp_ms:
            current_cue = SrtCue.merge(current_cue, cue)
            continue

        # extract current cue
        next_cue = None if idx + 1 >= len(cues) else cues[idx + 1]

        head_ms = max(
            current_cue._head_timestamp_ms - 2000,
            previous_ending_ms)

        tail_ms = min(
            current_cue._tail_timestamp_ms + 2000,
            current_cue._tail_timestamp_ms + 2000
            if next_cue is None else next_cue._head_timestamp_ms)

        extract_span([current_cue], source_mp3_file, source_hash,
                     target_dir_path, sample_rate, head_ms, tail_ms)

        previous_ending_ms = current_cue._tail_timestamp_ms

        current_cue = cue

    if current_cue is not None:
        head_ms = max(
            current_cue._head_timestamp_ms - 2000,
            previous_ending_ms)

        tail_ms = current_cue._tail_timestamp_ms + 2000

        extract_span([current_cue], source_mp3_file, source_hash,
                     target_dir_path, sample_rate, head_ms, tail_ms)


def prepare_training_data(params):
    """
    """
    # build name table
    target_names = {}

    target_temp = collect_file_names(params['path_target_training'], 'srt')

    for name in target_temp:
        name = os.path.split(name)[1]

        tags = name.split('_')

        if len(tags) > 1:
            target_names[tags[-2]] = None

    source_names = collect_file_names(params['path_source_training'], 'srt')

    for source_name in source_names:
        source_name = os.path.splitext(source_name)[0]
        source_hash = source_name[:-24]

        # processed?
        if source_hash in target_names:
            continue

        sample_rate = params['sample_rate']
        target_dir_path = params['path_target_training']
        source_dir_path = params['path_source_training']
        source_srt_file = source_dir_path + source_name + '.srt'
        source_mp3_file = source_dir_path + source_name + '.mp3'

        cues = load_cues(source_srt_file)

        extract_sound(
            cues, source_mp3_file, source_hash, target_dir_path, sample_rate)
        extract_voice(
            cues, source_mp3_file, source_hash, target_dir_path, sample_rate)


def prepare_test_data_x(path_source, path_target):
    """
    """
    names = collect_file_names(path_source, 'srt')

    for name in names:
        name_srt = name
        name_mp3 = os.path.splitext(name)[0] + '.mp3'
        name_wav = os.path.splitext(name)[0] + '.wav'

        # transformed before
        if os.path.isfile(os.path.join(path_target, name)):
            continue

        # clean srt
        old_cues = load_cues(path_source + name)
        new_cues = []

        current_cue = None

        for cue in old_cues:
            # NOTE: assume lines include '([<>])/' are invalid!!!
            if not is_valid_line(cue._line):
                continue

            if current_cue is None:
                current_cue = cue
                continue

            if current_cue._tail_timestamp_ms + 500 < cue._head_timestamp_ms:
                new_cues.append(current_cue)
                current_cue = cue
            else:
                current_cue = SrtCue.merge(current_cue, cue)

        if current_cue is not None:
            new_cues.append(current_cue)

        SrtWriter.save(path_target + name_srt, new_cues, 0)

        # mp3 to wav
        source_mp3 = path_source + name_mp3
        target_wav = path_target + name_wav

        command = 'ffmpeg -i {} -acodec pcm_f32le'.format(source_mp3)
        command += ' -ac 1 -ar {}'.format(params['sample_rate'])
        command += ' -map_metadata -1 -write_xing 0 {}'.format(target_wav)

        os.system(command)


def prepare_test_data(params):
    """
    """
    prepare_test_data_x(
        params['path_source_test'], params['path_target_test'])


def prepare_validate_data(params):
    """
    """
    prepare_test_data_x(
        params['path_source_validate'], params['path_target_validate'])


if __name__ == "__main__":
    params = prepare_params()

    prepare_training_data(params)
    prepare_test_data(params)
    prepare_validate_data(params)
