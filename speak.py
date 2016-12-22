"""
"""
import hashlib
import os
import numpy as np
import scipy.io.wavfile as wav


class Arguments(object):
    """
    """
    @staticmethod
    def make_dir(path):
        """
        """
        if not os.path.isdir(path):
            os.makedirs(path)

    def __init__(self, sample_rate=4000):
        """
        """
        self._voices = ["Alex", "Daniel", "Fiona", "Fred", "Karen", "Samantha",
                        "Tessa", "Victoria"]

        self._sample_rate = sample_rate

        rate_name = str(self._sample_rate)

        self._dir_root = './datasets/'
        self._dir_sentences = os.path.join(self._dir_root, 'sen')
        self._dir_voice = os.path.join(self._dir_root, 'voice')
        self._dir_voice_raw = os.path.join(self._dir_voice, 'raw')
        self._dir_voice_raw_test = os.path.join(self._dir_voice_raw, 'test')
        self._dir_voice_raw_training = os.path.join(
            self._dir_voice_raw, 'training')
        self._dir_voice_raw_validate = os.path.join(
            self._dir_voice_raw, 'validate')
        self._dir_voice_target = os.path.join(self._dir_voice, rate_name)
        self._dir_voice_target_test = os.path.join(
            self._dir_voice_target, 'test')
        self._dir_voice_target_training = os.path.join(
            self._dir_voice_target, 'training')
        self._dir_voice_target_validate = os.path.join(
            self._dir_voice_target, 'validate')
        self._dir_bground = os.path.join(self._dir_root, 'bg')
        self._dir_bground_raw = os.path.join(self._dir_bground, 'raw')
        self._dir_bground_target = os.path.join(self._dir_bground, rate_name)
        self._dir_bground_target_training = os.path.join(
            self._dir_bground_target, 'training')
        self._dir_bground_target_validate = os.path.join(
            self._dir_bground_target, 'validate')
        self._dir_bground_target_test = os.path.join(
            self._dir_bground_target, 'test')
        self._dir_cue = os.path.join(self._dir_root, 'cue')
        self._dir_cue_target = os.path.join(self._dir_cue, rate_name)
        self._dir_cue_target_training = os.path.join(
            self._dir_cue_target, 'training')
        self._dir_cue_target_validate = os.path.join(
            self._dir_cue_target, 'validate')
        self._dir_cue_target_test = os.path.join(
            self._dir_cue_target, 'test')

        self._path_transcripts = os.path.join(self._dir_root, 'transcript.txt')

        if not os.path.isdir(self._dir_root):
            raise Exception('invalid root path')

        if not os.path.isfile(self._path_transcripts):
            raise Exception('need the transcript')

        Arguments.make_dir(self._dir_sentences)
        Arguments.make_dir(self._dir_voice)
        Arguments.make_dir(self._dir_voice_raw)
        Arguments.make_dir(self._dir_voice_raw_test)
        Arguments.make_dir(self._dir_voice_raw_training)
        Arguments.make_dir(self._dir_voice_raw_validate)
        Arguments.make_dir(self._dir_voice_target)
        Arguments.make_dir(self._dir_voice_target_test)
        Arguments.make_dir(self._dir_voice_target_training)
        Arguments.make_dir(self._dir_voice_target_validate)
        Arguments.make_dir(self._dir_bground)
        Arguments.make_dir(self._dir_bground_raw)
        Arguments.make_dir(self._dir_bground_target)
        Arguments.make_dir(self._dir_bground_target_test)
        Arguments.make_dir(self._dir_bground_target_training)
        Arguments.make_dir(self._dir_bground_target_validate)
        Arguments.make_dir(self._dir_cue)
        Arguments.make_dir(self._dir_cue_target)
        Arguments.make_dir(self._dir_cue_target_test)
        Arguments.make_dir(self._dir_cue_target_training)
        Arguments.make_dir(self._dir_cue_target_validate)


def collect_file_names(path_dir, extension):
    """
    """
    names = []

    for name in os.listdir(path_dir):
        if not os.path.isfile(os.path.join(path_dir, name)):
            continue

        if not name.endswith(extension):
            continue

        names.append(name)

    return names


def make_timestamp(idx, sample_rate):
    """
    """
    t = idx * 1000 / sample_rate

    H, t = t / 3600000, t % 3600000
    M, t = t / 60000, t % 60000
    S, m = t / 1000, t % 1000

    return "{}:{}:{},{}".format(H, M, S, m)


def is_name_for_test(name):
    """
    """
    return name[0] == 'f'


def is_name_for_validate(name):
    """
    """
    return name[0] == 'b'


def arrange_sentences(args):
    """
    """
    print 'generating sentences'

    article = [line.strip() for line in open(args._path_transcripts)]
    article = "".join(article)
    article = article.split('.')[:-1]

    for line in article:
        line = line.strip().lower() + '.'

        if len(line) == 0:
            continue

        name = hashlib.md5(line).hexdigest() + '.txt'
        path = os.path.join(args._dir_sentences, name)

        if os.path.isfile(path):
            continue

        with open(path, 'w') as output:
            output.write(line)


def speak(args):
    """
    """
    print 'speaking'

    path_source = args._dir_sentences

    sentences = []

    for f in os.listdir(path_source):
        if not os.path.isfile(os.path.join(path_source, f)):
            continue

        if not f.endswith('.txt'):
            continue

        sentences.append(f)

    for sentence in sentences:
        name = sentence[:-4]

        for voice in args._voices:
            target_name = hashlib.md5(name + voice).hexdigest() + '.caf'

            source_path = os.path.join(args._dir_sentences, sentence)

            if not os.path.isfile(source_path):
                continue

            if is_name_for_validate(target_name):
                target_dir = args._dir_voice_raw_validate
            elif is_name_for_test(target_name):
                target_dir = args._dir_voice_raw_test
            else:
                target_dir = args._dir_voice_raw_training

            target_path = os.path.join(target_dir, target_name)

            if os.path.isfile(target_path):
                continue

            command_say = "/usr/bin/say -v {} -f {} -o {}".format(
                    voice, source_path, target_path)

            os.system(command_say)


def resample_voices(args):
    """
    """
    dir_pairs = [
        (args._dir_voice_raw_test, args._dir_voice_target_test),
        (args._dir_voice_raw_training, args._dir_voice_target_training),
        (args._dir_voice_raw_validate, args._dir_voice_target_validate)]

    for dirs in dir_pairs:
        source_dir, target_dir = dirs[0], dirs[1]

        print 'resampling voices: {} -> {}'.format(source_dir, target_dir)

        for caf in os.listdir(source_dir):
            if not os.path.isfile(os.path.join(source_dir, caf)):
                continue

            if not caf.endswith('.caf'):
                continue

            name = caf[:-4]

            source_path = os.path.join(source_dir, caf)
            target_path = os.path.join(target_dir, name + '.wav')

            if os.path.isfile(target_path):
                continue

            command = "ffmpeg -i {} -acodec pcm_f32le -ac 1 -ar {} {}" \
                .format(source_path, args._sample_rate, target_path)

            os.system(command)


def resample_bg(args):
    """
    """
    print 're-sampling background sound'

    source_dir = args._dir_bground_raw

    for mp3 in os.listdir(source_dir):
        if not os.path.isfile(os.path.join(source_dir, mp3)):
            continue

        if not mp3.endswith('.mp3'):
            continue

        name = mp3[:-4]

        if is_name_for_validate(mp3):
            target_dir = args._dir_bground_target_validate
        elif is_name_for_test(mp3):
            target_dir = args._dir_bground_target_test
        else:
            target_dir = args._dir_bground_target_training

        source_path = os.path.join(source_dir, mp3)
        target_path = os.path.join(target_dir, name + '.wav')

        if os.path.isfile(target_path):
            continue

        command = "ffmpeg -i {} -acodec pcm_f32le -ac 1 -ar {} {}" \
            .format(source_path, args._sample_rate, target_path)

        os.system(command)


def mix(args):
    """
    """
    categories = [[], [], []]

    categories[0].append(args._dir_bground_target_training)
    categories[0].append(args._dir_voice_target_training)
    categories[0].append(args._dir_cue_target_training)
    categories[1].append(args._dir_bground_target_validate)
    categories[1].append(args._dir_voice_target_validate)
    categories[1].append(args._dir_cue_target_validate)
    categories[2].append(args._dir_bground_target_test)
    categories[2].append(args._dir_voice_target_test)
    categories[2].append(args._dir_cue_target_test)

    for category in categories:
        sound_dir, voice_dir, movie_dir = category

        bgm_names = collect_file_names(sound_dir, 'wav')
        wav_names = collect_file_names(voice_dir, 'wav')

        print 'mix category: {}'.format(sound_dir)

        for bgm_name in bgm_names:
            _, bgm = wav.read(os.path.join(sound_dir, bgm_name))

            index = 0
            timestamps = []

            wav_name_indice = np.random.choice(
                len(wav_names), [len(wav_names)], replace=False)

            for wav_name_index in wav_name_indice:
                wav_name = wav_names[wav_name_index]

                _, tmp = wav.read(os.path.join(voice_dir, wav_name))

                if index + len(tmp) >= len(bgm):
                    break

                factor = 0.5 * np.random.random()

                for x in xrange(len(tmp)):
                    bgm[index + x] = bgm[index + x] * factor + tmp[x]

                timestamps.append((index, len(tmp)))

                index += 2 * len(tmp)

            mov_name = hashlib.md5(bgm).hexdigest()

            target_wav = os.path.join(movie_dir, mov_name + '.wav')
            target_srt = os.path.join(movie_dir, mov_name + '.srt')

            wav.write(target_wav, args._sample_rate, bgm)

            # srt
            with open(target_srt, 'w') as srt:
                for x in xrange(len(timestamps)):
                    time_head = timestamps[x][0]
                    time_tail = timestamps[x][1] + time_head
                    time_head = make_timestamp(time_head, args._sample_rate)
                    time_tail = make_timestamp(time_tail, args._sample_rate)

                    srt.write("{}\n".format(x))
                    srt.write("{} --> {}\n".format(time_head, time_tail))
                    srt.write("!@#$%^\n\n")


if __name__ == "__main__":
    args = Arguments(4000)

    arrange_sentences(args)
    speak(args)
    resample_voices(args)
    resample_bg(args)
    mix(args)
