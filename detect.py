"""
"""
import os

from model import VadModel
from parameters import Parameters
from srt_features import SrtWriter
from wav_features import WavFeatures


params = Parameters()

source_wav = params.get_movie_path()
target_srt = os.path.splitext(source_wav)[0] + '.srt'

batch_size = 1
sequence_size = params.get_rnn_sequence_length()
wav_window_size = params.get_wav_window_size()
wav_window_step = params.get_wav_window_step()
wav_sample_rate = params.get_wav_sample_rate()

wav_window_size_second = float(wav_window_size) / float(wav_sample_rate)
wav_window_step_second = float(wav_window_step) / float(wav_sample_rate)

input_wav = WavFeatures()

input_wav.load(
    source_wav,
    feature_type=params.get_wav_feature_type(),
    window_size=wav_window_size_second,
    window_step=wav_window_step_second,
    numcep=params.get_wav_cepstrum_size(),
    feature_mean=params.get_wav_feature_mean(),
    feature_std=params.get_wav_feature_std())

wav_features = input_wav.features()

wav_batch = [wav_features]

model = VadModel(params)

features = model.detect(wav_batch)

SrtWriter.save(target_srt, features[0], wav_sample_rate, wav_window_step)
