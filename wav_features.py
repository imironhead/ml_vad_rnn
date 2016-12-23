"""
"""
import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import mfcc


class WavFeatures(object):
    """
    """
    def __init__(self):
        """
        """
        self._features = None
        self._sample_rate = None
        self._window_size = None
        self._window_step = None

    def load(self, path, window_size=0.025, window_step=0.01, numcep=13):
        """
        """
        self._window_size = window_size
        self._window_step = window_step

        self._sample_rate, self._features = wav.read(path)

        if len(self._features.shape) > 1:
            self._features = self._features[:, 0]

        self._features = mfcc(
            self._features,
            self._sample_rate,
            winlen=window_size,
            winstep=window_step,
            numcep=numcep)

    def feature_shape(self):
        """
        """
        return self._features[0].shape

    def feature_size(self):
        """
        """
        return len(self._features)

    def features(self, begin, end, padding):
        """
        """
        padding_len = padding
        feature_len = end - begin
        feature_dim = self._features[0].shape[0]

        output = np.zeros((feature_len + padding_len, feature_dim))

        output[0:feature_len] = self._features[begin:end]

        return output

    def sample_rate(self):
        """
        """
        return self._sample_rate

    def window_size(self):
        """
        """
        return self._window_size

    def window_step(self):
        """
        """
        return self._window_step
