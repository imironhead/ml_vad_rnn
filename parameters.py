"""
"""
import json
import os
import sys


class Parameters(object):
    """
    """
    @staticmethod
    def make_dir(dir_path):
        """
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def arg():
        """
        """
        if len(sys.argv) != 2:
            raise Exception('need parameter file name')

        if not os.path.isfile(sys.argv[1]):
            raise Exception('invalid parameter file path')

        _, name = os.path.split(sys.argv[1])

        param = {}

        with open(sys.argv[1]) as source:
            param = json.load(source)

        if name != 'param_{}.json'.format(param['session_name']):
            raise Exception('invalid session name')

        return param

    def __init__(self):
        """
        """
        param = Parameters.arg()

        self._session_name = param['session_name']
        self._model_name = param['model_name']
        self._epoch_size = param['epoch_size']
        self._epoch_count = param['epoch_count']
        self._rnn_cell = param['rnn_cell']
        self._rnn_unit_num = param['rnn_unit_num']
        self._rnn_sequence_length = param['rnn_sequence_length']
        self._batch_size = param['batch_size']
        self._optimizer = param['optimizer']
        self._learning_rate = param['learning_rate']

        self._wav_sample_rate = param['wav_sample_rate']
        self._wav_feature = param['wav_feature']
        self._wav_cepstrum_size = param['wav_cepstrum_size']
        self._wav_window_size = param['wav_window_size']
        self._wav_window_step = param['wav_window_step']

        self._srt_delay_size = param['srt_delay_size']

        self._head_hidden_layers_bias = param['head_hidden_layers_bias']
        self._tail_hidden_layers_bias = param['tail_hidden_layers_bias']
        self._head_hidden_layers_nonlinear = \
            param['head_hidden_layers_nonlinear']
        self._tail_hidden_layers_nonlinear = \
            param['tail_hidden_layers_nonlinear']

        self._nn_hidden_layer_before_rnn = param['head_hidden_layers']
        self._nn_hidden_layer_after_rnn = param['tail_hidden_layers']

        self._checkpoint_source_path = './checkpoints/'
        self._checkpoint_target_path = './checkpoints/model_{}.ckpt'.format(
            self._session_name)
        self._tensorboard_log_root = './tensorboards/'
        self._tensorboard_log_path = './tensorboards/{}/'.format(
            self._session_name)

        Parameters.make_dir(self._checkpoint_source_path)
        Parameters.make_dir(self._checkpoint_target_path)
        Parameters.make_dir(self._tensorboard_log_root)
        Parameters.make_dir(self._tensorboard_log_path)

        self._dir_cue_training = './datasets/cue/{}/training/'.format(
            self._wav_sample_rate)
        self._dir_cue_validate = './datasets/cue/{}/validate/'.format(
            self._wav_sample_rate)
        self._dir_cue_test = './datasets/cue/{}/test/'.format(
            self._wav_sample_rate)

        if not os.path.isdir(self._dir_cue_training):
            raise Exception('need training dir')
        if not os.path.isdir(self._dir_cue_validate):
            raise Exception('need validate dir')
        if not os.path.isdir(self._dir_cue_test):
            raise Exception('need test dir')

    def get_session_name(self):
        """
        """
        return self._session_name

    def get_model_name(self):
        """
        """
        return self._model_name

    def get_epoch_size(self):
        """
        """
        return self._epoch_size

    def get_epoch_count(self):
        """
        """
        return self._epoch_count

    def get_batch_size(self):
        """
        """
        return self._batch_size

    def get_optimizer(self):
        """
        """
        return str(self._optimizer)

    def get_learning_rate(self):
        """
        """
        return self._learning_rate

    def get_rnn_cell(self):
        """
        """
        return self._rnn_cell

    def get_rnn_unit_num(self):
        """
        """
        return self._rnn_unit_num

    def get_rnn_sequence_length(self):
        """
        """
        return self._rnn_sequence_length

    def get_wav_sample_rate(self):
        """
        """
        return self._wav_sample_rate

    def get_wav_feature(self):
        """
        """
        return str(self._wav_feature)

    def get_wav_cepstrum_size(self):
        """
        """
        return self._wav_cepstrum_size

    def get_wav_window_size(self):
        """
        """
        return self._wav_window_size

    def get_wav_window_step(self):
        """
        """
        return self._wav_window_step

    def get_srt_delay_size(self):
        """
        """
        return self._srt_delay_size

    def get_non_linear_gate_before_rnn(self):
        """
        """
        return str(self._head_hidden_layers_nonlinear)

    def get_non_linear_gate_after_rnn(self):
        """
        """
        return str(self._tail_hidden_layers_nonlinear)

    def get_hidden_layer_dim_before_rnn(self):
        """
        """
        return list(self._nn_hidden_layer_before_rnn)

    def get_hidden_layer_dim_after_rnn(self):
        """
        """
        return list(self._nn_hidden_layer_after_rnn)

    def get_checkpoint_source_path(self):
        """
        """
        return self._checkpoint_source_path

    def get_checkpoint_target_path(self):
        """
        """
        return self._checkpoint_target_path

    def get_tensorboard_log_path(self):
        """
        """
        return self._tensorboard_log_path

    def get_dir_cue_training(self):
        """
        """
        return self._dir_cue_training

    def get_dir_cue_validate(self):
        """
        """
        return self._dir_cue_validate

    def get_dir_cue_test(self):
        """
        """
        return self._dir_cue_test

    def should_use_adam(self):
        """
        """
        return self._optimizer == 'adam'

    def should_use_lstm(self):
        """
        """
        return self._rnn_cell == 'lstm'

    def should_use_mfcc(self):
        """
        """
        return self._wav_feature == 'mfcc'

    def should_add_bias_before_rnn(self):
        """
        """
        return self._head_hidden_layers_bias

    def should_add_bias_after_rnn(self):
        """
        """
        return self._tail_hidden_layers_bias

    def should_use_relu_before_rnn(self):
        """
        """
        return self._head_hidden_layers_nonlinear == 'relu'

    def should_use_relu_after_rnn(self):
        """
        """
        return self._tail_hidden_layers_nonlinear == 'relu'