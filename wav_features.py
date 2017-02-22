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

    def load(self,
             path,
             feature_type='mfcc',
             window_size=0.025,
             window_step=0.01,
             numcep=13,
             feature_mean=np.zeros((13)),
             feature_std=np.ones((13))):
        """
        """
        self._window_size = window_size
        self._window_step = window_step

        self._sample_rate, self._features = wav.read(path)

        if len(self._features.shape) > 1:
            self._features = self._features[:, 0]

        if feature_type == 'mfcc':
            self._features = mfcc(
                self._features,
                self._sample_rate,
                winlen=window_size,
                winstep=window_step,
                numcep=numcep)
        else:
            # 'raw'
            w_step = int(self._sample_rate * window_step)
            w_size = numcep
            d_size = len(self._features)

            self._features = [self._features[i:i+w_size]
                              for i in xrange(0, d_size - w_size + 1, w_step)]

            self._features = np.vstack(self._features)

        self._features -= feature_mean
        self._features /= feature_std

    def feature_shape(self):
        """
        """
        return self._features[0].shape

    def feature_size(self):
        """
        """
        return len(self._features)

    def features(self):
        """
        """
        return self._features

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
