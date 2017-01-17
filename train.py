"""
"""
import os
import numpy as np
from model import VadModel
from parameters import Parameters
from srt_features import SrtFeatures
from wav_features import WavFeatures


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


def train(params, model, context):
    """
    """
    data_dir_training = params.get_dir_cue_training()
    data_wav_training = collect_file_names(data_dir_training, 'wav')

    batch_size = params.get_batch_size()
    sequence_size = params.get_rnn_sequence_length()
    wav_window_size = params.get_wav_window_size()
    wav_window_step = params.get_wav_window_step()
    wav_sample_rate = params.get_wav_sample_rate()

    wav_window_size_second = float(wav_window_size) / float(wav_sample_rate)
    wav_window_step_second = float(wav_window_step) / float(wav_sample_rate)

    wav_batch = []
    srt_batch = []

    for i in xrange(batch_size):
        while True:
            wav_index = np.random.randint(0, len(data_wav_training))

            name, _ = os.path.splitext(data_wav_training[wav_index])

            data_wav_path = os.path.join(data_dir_training, name + '.wav')
            data_srt_path = os.path.join(data_dir_training, name + '.srt')

            train_wav = WavFeatures()
            train_srt = SrtFeatures()

            train_wav.load(
                data_wav_path,
                window_size=wav_window_size_second,
                window_step=wav_window_step_second,
                numcep=params.get_wav_cepstrum_size())
            train_srt.load(data_srt_path, wav_sample_rate)

            if train_wav.feature_size() > sequence_size:
                break

        wav_features = train_wav.features()
        srt_features = train_srt.features(
            0, len(wav_features), wav_window_size, wav_window_step)

        wav_batch.append(wav_features)
        srt_batch.append(srt_features)

    _, gstep, summaries, loss, accuracy, _ = model.train(wav_batch, srt_batch)

    if context['gstep'] + 100 < gstep:
        context['gstep'] = gstep

        model.save_summary(gstep, summary=summaries)

        print "step:{}, loss: {}, accuracy: {}".format(gstep, loss, accuracy)

    return gstep


def test(params, model, context):
    """
    """
    data_dir_test = params.get_dir_cue_test()
    data_wav_test = collect_file_names(data_dir_test, 'wav')

    sequence_size = params.get_rnn_sequence_length()
    wav_window_size = params.get_wav_window_size()
    wav_window_step = params.get_wav_window_step()
    wav_sample_rate = params.get_wav_sample_rate()

    wav_window_size_second = float(wav_window_size) / float(wav_sample_rate)
    wav_window_step_second = float(wav_window_step) / float(wav_sample_rate)

    wav_batch = []
    srt_batch = []

    wav_index = np.random.randint(0, len(data_wav_test))

    name, _ = os.path.splitext(data_wav_test[wav_index])

    data_wav_path = os.path.join(data_dir_test, name + '.wav')
    data_srt_path = os.path.join(data_dir_test, name + '.srt')

    train_wav = WavFeatures()
    train_srt = SrtFeatures()

    train_wav.load(
        data_wav_path,
        window_size=wav_window_size_second,
        window_step=wav_window_step_second,
        numcep=params.get_wav_cepstrum_size())
    train_srt.load(data_srt_path, wav_sample_rate)

    wav_features = train_wav.features()
    srt_features = train_srt.features(
        0, len(wav_features), wav_window_size, wav_window_step)

    wav_batch.append(wav_features)
    srt_batch.append(srt_features)

    gstep, accuracy = model.test(wav_batch, srt_batch)

    model.save_summary(gstep, "test accuracy", accuracy)

    print 'step:{}, test accuracy: {}'.format(gstep, accuracy)


def learn(params, model):
    """
    """
    context = {
        'gstep': 0,
    }

    for epoch in xrange(params.get_epoch_count()):
        for step in xrange(params.get_epoch_size()):
            gstep = train(params, model, context)

            if gstep >= params.get_max_steps():
                return

        test(params, model, context)


if __name__ == "__main__":
    """
    """
    params = Parameters()

    model = VadModel(params)

    try:
        learn(params, model)
    except KeyboardInterrupt:
        pass

    model.save_checkpoint()
