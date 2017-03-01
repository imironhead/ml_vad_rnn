"""
mix background sounds (environment) and foreground sounds (speech) to generate
training dada.
"""
import numpy as np
import os
import scipy.io.wavfile as wav
import sys

from util import make_dir, md5
from util import background_raw_sound_path, foreground_raw_sound_path
from util import mix_sound_path


class ForegroundSound(object):
    """
    foreground sound 'generator'.
    """
    def __init__(self, path_dir, sample_rate):
        """
        """
        self._sample_rate = sample_rate
        self._wav_paths = [
            os.path.join(path_dir, n)
            for n in os.listdir(path_dir) if n.endswith('.wav')]

    def next_span(self):
        """
        return a speech from a random wav file.
        """
        i = np.random.randint(len(self._wav_paths))

        wav_sample_rate, samples = wav.read(self._wav_paths[i])

        # sanity check. only accept specified wav with sampling rate.
        if wav_sample_rate != self._sample_rate:
            raise Exception('something wrong, invalid sample rate')

        # sanity check. only accept mono channel wavs.
        if len(samples.shape) != 1:
            samples = samples[:, 0]

        return samples


class BackgroundSound(object):
    """
    background sound 'generator'.
    """
    def __init__(self, path_dir, sample_rate):
        """
        """
        self._sample_rate = sample_rate
        self._wav_buffer = np.zeros((0))
        self._wav_paths = [
            os.path.join(path_dir, n)
            for n in os.listdir(path_dir) if n.endswith('.wav')]

    def next_span(self, duration_second=10):
        """
        return a background sound array in duration_second long.
        """
        duration_hz = duration_second * self._sample_rate

        result = np.zeros((duration_hz))

        index = 0

        while index < duration_hz:
            if self._wav_buffer.shape[0] == 0:
                i = np.random.randint(len(self._wav_paths))

                sample_rate, self._wav_buffer = wav.read(self._wav_paths[i])

                # sanity check. only accept wavs with specified sampling rate.
                if sample_rate != self._sample_rate:
                    raise Exception('something wrong, invalid sample rate')

                # sanity check. use mono channel.
                if len(self._wav_buffer.shape) > 1:
                    self._wav_buffer = self._wav_buffer[:, 0]

                # random scaling to produce more diversity.
                self._wav_buffer *= 0.05 + 0.95 * np.random.random()

            l = min(self._wav_buffer.shape[0], duration_hz - index)

            result[index:index + l] = self._wav_buffer[:l]

            self._wav_buffer = self._wav_buffer[l:]

            index += l

        return result


def make_timestamp(idx, sample_rate):
    """
    make timestamp for srt.
    'idx': position in hz.
    'sample_rate': hz
    """
    t = idx * 1000 / sample_rate

    H, t = t / 3600000, t % 3600000
    M, t = t / 60000, t % 60000
    S, m = t / 1000, t % 1000

    return "{}:{}:{},{}".format(H, M, S, m)


def mix(target,
        sample_rate,
        duration_segment_seconds,
        generator_bgs,
        generator_fgs):
    """
    mixing and saving a training wav and its' srt.
    """
    timestamps = []

    bgs = generator_bgs.next_span(duration_segment_seconds)

    # mix without speech for 2% of training data.
    if np.random.random() < 0.98:
        b = sample_rate
        e = duration_segment_seconds * sample_rate

        while b + sample_rate < e:
            w_f = 0.1 * np.random.random() + 0.9
            w_b = 0.1 * np.random.random()

            bgf = generator_fgs.next_span()

            l = min(e - b, bgf.shape[0])

            bgs[b:b+l] = w_b * bgs[b:b+l] + w_f * bgf[:l]

            timestamps.append((b, l))

            # advance for speech
            b = b + l

            # diverse gap
            f = 0.5 + 1.5 * np.random.random()

            # advance for gap
            b = b + int(f * sample_rate)

    # we do not know md5 of the wav, use a temp path instead.
    path_tmp = os.path.join(target, 'temp.wav')

    wav.write(path_tmp, sample_rate, bgs)

    mix_name = md5(path_tmp)

    path_wav = os.path.join(target, mix_name + '.wav')
    path_srt = os.path.join(target, mix_name + '.srt')

    # change the name of the wav back to md5 format.
    os.system('mv {} {}'.format(path_tmp, path_wav))

    # output the srt.
    with open(path_srt, 'w') as srt:
        for x in xrange(len(timestamps)):
            time_head = timestamps[x][0]
            time_tail = timestamps[x][1] + time_head
            time_head = make_timestamp(time_head, sample_rate)
            time_tail = make_timestamp(time_tail, sample_rate)

            srt.write("{}\n".format(x))
            srt.write("{} --> {}\n".format(time_head, time_tail))
            srt.write("!@#$%^\n\n")


def main():
    """
    """
    sample_rate = \
        int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    duration_total = \
        int(sys.argv[2]) if len(sys.argv) > 2 else 60 * 60
    duration_segment = \
        int(sys.argv[3]) if len(sys.argv) > 3 else 10

    source_bgs = background_raw_sound_path(sample_rate)
    source_fgs = foreground_raw_sound_path(sample_rate)
    target_mix = mix_sound_path(sample_rate)

    if not os.path.isdir(source_bgs) or not os.path.isdir(source_fgs):
        raise Exception('found no source directories')

    make_dir(target_mix)

    g_bgs = BackgroundSound(source_bgs, sample_rate)
    g_fgs = ForegroundSound(source_fgs, sample_rate)

    for t in xrange(0, duration_total, duration_segment):
        mix(target_mix, sample_rate, duration_segment, g_bgs, g_fgs)

        if (t / duration_segment) % 100 == 0:
            print 'progress: {} / {}'.format(t, duration_total)


if __name__ == '__main__':
    main()
