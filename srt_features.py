"""
"""
import numpy as np


class SrtCue(object):
    """
    """
    def __init__(self, lines, sample_rate):
        """
        """
        self._head_timestamp = None
        self._tail_timestamp = None

        if not isinstance(lines, list) or len(lines) < 3:
            return

        b, e = lines[1].split('-->')

        self._head_timestamp = SrtCue.timestamp_to_timescale(b, sample_rate)
        self._tail_timestamp = SrtCue.timestamp_to_timescale(e, sample_rate)

    @staticmethod
    def timestamp_to_ms(timestamp):
        """
        """
        h, m, s, ms = timestamp.replace(',', ':').split(':')

        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    @staticmethod
    def ms_to_timescale(ms, timescale):
        """
        """
        return ms * timescale / 1000

    @staticmethod
    def timestamp_to_timescale(timestamp, timescale):
        ms = SrtCue.timestamp_to_ms(timestamp)
        return SrtCue.ms_to_timescale(ms, timescale)

    def is_valid(self):
        return self._head_timestamp_ms is not None


class SrtFeatures(object):
    """
    """
    def __init__(self):
        """
        """
        self._cues = []

    def cues(self):
        """
        """
        for cue in self._cues:
            yield cue

    def load(self, path, sample_rate):
        """
        """
        lines = []

        for line in open(path):
            line = line.rstrip()

            if len(line) == 0:
                self._cues.append(SrtCue(lines, sample_rate))

                lines = []
            else:
                lines.append(line)

        # last cue
        if len(lines) > 0:
            self._cues.append(SrtCue(lines, sample_rate))

    def features(self, begin, end, delay, window_size, window_step):
        """
        """
        source_len = len(self._cues)
        target_len = end - begin

        source_idx = 0
        target_idx = delay

        result = np.zeros([delay + target_len], dtype=np.int32)

        while source_idx < source_len and target_idx < target_len:
            target_b = window_step * (begin + target_idx)
            target_e = window_size + target_b

            cue = self._cues[source_idx]

            if cue._tail_timestamp <= target_b:
                source_idx += 1
                continue

            if cue._head_timestamp >= target_e:
                target_idx += 1
                continue

            result[target_idx] = 1

            target_idx += 1

        return result


class SrtWriter(object):
    @staticmethod
    def timestamp(idx, step, sample_rate):
        """
        """
        t = idx * step * 1000 / sample_rate

        H, t = t / 3600000, t % 3600000
        M, t = t / 60000, t % 60000
        S, m = t / 1000, t % 1000

        return "{}:{}:{},{}".format(H, M, S, m)

    @staticmethod
    def save(path, features, sample_rate, window_step):
        """
        """
        with open(path, 'w') as srt:
            caption_idx = 0
            feature_idx = 0

            while True:
                while True:
                    if feature_idx >= len(features):
                        break
                    if features[feature_idx] == 1:
                        break
                    feature_idx += 1

                if feature_idx >= len(features):
                    break

                time_head = \
                    SrtWriter.timestamp(feature_idx, window_step, sample_rate)

                while True:
                    if feature_idx >= len(features):
                        break
                    if features[feature_idx] == 0:
                        break
                    feature_idx += 1

                time_tail = \
                    SrtWriter.timestamp(feature_idx, window_step, sample_rate)

                srt.write("{}\n".format(caption_idx))
                srt.write("{} --> {}\n".format(time_head, time_tail))
                srt.write("!@#$%^\n\n")

                caption_idx += 1
