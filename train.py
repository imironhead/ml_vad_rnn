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


def test(params, model):
    """
    """
    data_dir_test = params.get_dir_cue_test()
    data_wav_test = collect_file_names(data_dir_test, 'wav')

    batch_size = 200
    sample_size = 200
    srt_delay_size = params.get_srt_delay_size()
    training_sequence_size = params.get_rnn_sequence_length()
    wav_window_size = params.get_wav_window_size()
    wav_window_step = params.get_wav_window_step()
    wav_sample_rate = params.get_wav_sample_rate()

    wav_window_size_second = float(wav_window_size) / float(wav_sample_rate)
    wav_window_step_second = float(wav_window_step) / float(wav_sample_rate)

    accuracy_total = 0.0
    accuracy_count = 0.0

    for _ in xrange(0, sample_size, batch_size):
        choice_replace = len(data_wav_test) < batch_size

        indice = np.random.choice(
            len(data_wav_test), [batch_size], replace=choice_replace)

        wav_batch = [[] for x in xrange(batch_size)]
        srt_batch = [[] for x in xrange(batch_size)]

        for i in xrange(len(indice)):
            name, _ = os.path.splitext(data_wav_test[indice[i]])

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

            data_size = train_wav.feature_size() - training_sequence_size

            head = np.random.randint(0, data_size)
            tail = head + training_sequence_size - srt_delay_size

            wav_batch[i] = train_wav.features(head, tail, srt_delay_size)
            srt_batch[i] = train_srt.features(
                head, tail, srt_delay_size, wav_window_size, wav_window_step)

        gstep, summaries, loss, accuracy = model.test(wav_batch, srt_batch)

        accuracy_total += accuracy
        accuracy_count += 1.0

    accuracy = accuracy_total / accuracy_count

    model.save_summary(gstep, "test accuracy", accuracy)

    print 'step:{}, test accuracy: {}'.format(gstep, accuracy)


def train(params, model):
    """
    """
    data_dir_training = params.get_dir_cue_training()
    data_wav_training = collect_file_names(data_dir_training, 'wav')

    batch_size = params.get_batch_size()
    srt_delay_size = params.get_srt_delay_size()
    training_sequence_size = params.get_rnn_sequence_length()
    wav_window_size = params.get_wav_window_size()
    wav_window_step = params.get_wav_window_step()
    wav_sample_rate = params.get_wav_sample_rate()

    wav_window_size_second = float(wav_window_size) / float(wav_sample_rate)
    wav_window_step_second = float(wav_window_step) / float(wav_sample_rate)

    choice_replace = len(data_wav_training) < batch_size

    indice = np.random.choice(
        len(data_wav_training), [batch_size], replace=choice_replace)

    wav_batch = [[] for x in xrange(batch_size)]
    srt_batch = [[] for x in xrange(batch_size)]

    for i in xrange(len(indice)):
        name, _ = os.path.splitext(data_wav_training[indice[i]])

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

        data_size = train_wav.feature_size() - training_sequence_size

        head = np.random.randint(0, data_size)
        tail = head + training_sequence_size - srt_delay_size

        wav_batch[i] = train_wav.features(head, tail, srt_delay_size)
        srt_batch[i] = train_srt.features(
            head, tail, srt_delay_size, wav_window_size, wav_window_step)

    gstep, summaries, loss, accuracy = model.train(wav_batch, srt_batch)

    if gstep % 100 == 0:
        model.save_summary(gstep, summary=summaries)

        print "step:{}, loss: {}, accuracy: {}".format(gstep, loss, accuracy)

    return gstep


def learn(params, model):
    """
    """
    for epoch in xrange(params.get_epoch_count()):
        for step in xrange(params.get_epoch_size()):
            gstep = train(params, model)

            if gstep >= params.get_max_steps():
                return

        test(params, model)


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
