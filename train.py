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


def train(params, model):
    """
    """
    data_dir_training = params.get_dir_cue_training()
    data_wav_training = collect_file_names(data_dir_training, 'wav')

    data_dir_test = params.get_dir_cue_test()
    data_wav_test = collect_file_names(data_dir_test, 'wav')

    for epoch in xrange(params.get_epoch_count()):
        idx_training_data = np.random.randint(0, len(data_wav_training))

        print "training data: " + data_wav_training[idx_training_data]

        name, _ = os.path.splitext(data_wav_training[idx_training_data])

        data_wav_path = os.path.join(data_dir_training, name + '.wav')
        data_srt_path = os.path.join(data_dir_training, name + '.srt')

        train_wav = WavFeatures()
        train_srt = SrtFeatures()

        train_wav.load(data_wav_path, numcep=params.get_wav_cepstrum_size())
        train_srt.load(data_srt_path, params.get_wav_sample_rate())

        for step in xrange(params.get_epoch_size()):
            batch_size = params.get_batch_size()
            training_sequence_size = params.get_rnn_sequence_length()
            srt_delay_size = params.get_srt_delay_size()
            wav_window_size = params.get_wav_window_size()
            wav_window_step = params.get_wav_window_step()

            # build chunks
            wav_batch = [[] for x in xrange(batch_size)]
            srt_batch = [[] for x in xrange(batch_size)]

            data_size = train_wav.feature_size() - training_sequence_size

            indice = np.random.choice(data_size, [batch_size], replace=False)

            for i in xrange(len(indice)):
                idx = indice[i]
                head = idx
                tail = idx + training_sequence_size - srt_delay_size

                wav_batch[i] = train_wav.features(
                    head, tail, srt_delay_size)
                srt_batch[i] = train_srt.features(
                    head, tail, srt_delay_size,
                    wav_window_size, wav_window_step)

            loss, accc, gstep = model.train(wav_batch, srt_batch)

            if gstep % 100 == 0:
                print "step:{}, loss: {}, accc: {}".format(gstep, loss, accc)

            if gstep >= params.get_max_steps():
                return

        # test set
        idx_test_data = np.random.randint(0, len(data_wav_test))

        print "test data: " + data_wav_test[idx_test_data]

        name, _ = os.path.splitext(data_wav_test[idx_test_data])

        data_wav_path = os.path.join(data_dir_test, name + '.wav')
        data_srt_path = os.path.join(data_dir_test, name + '.srt')

        test_wav = WavFeatures()
        test_srt = SrtFeatures()

        test_wav.load(data_wav_path, numcep=params.get_wav_cepstrum_size())
        test_srt.load(data_srt_path, params.get_wav_sample_rate())

        head = 0
        tail = test_wav.feature_size()

        wav_batch = test_wav.features(head, tail, srt_delay_size)
        srt_batch = test_srt.features(
            head, tail, srt_delay_size,
            wav_window_size, wav_window_step)

        model.test(wav_batch, srt_batch)


if __name__ == "__main__":
    """
    """
    params = Parameters()

    model = VadModel(params)

    try:
        train(params, model)
    except KeyboardInterrupt:
        pass

    model.save_checkpoint()
