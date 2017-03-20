"""
"""
import json
import numpy as np
import os
import sys
import tensorflow as tf


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
        if len(sys.argv) < 2:
            raise Exception('need parameter file name')

        if not os.path.isfile(sys.argv[1]):
            raise Exception('invalid parameter file path')

        _, name = os.path.split(sys.argv[1])

        param = {}

        with open(sys.argv[1]) as source:
            param = json.load(source)

        if name != 'param_{}.json'.format(param['session_name']):
            raise Exception('invalid session name')

        if len(sys.argv) == 3:
            if not os.path.isfile(sys.argv[2]):
                raise Exception('invalid movie file path')

            param['movie_path'] = sys.argv[2]

        return param

    @staticmethod
    def leaky_relu(x, leak=0.2, name="lrelu"):
        """
        https://github.com/tensorflow/tensorflow/issues/4079
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1.0 + leak)
            f2 = 0.5 * (1.0 - leak)
            return f1 * x + f2 * abs(x)

    def __init__(self):
        """
        """
        self._params = param = Parameters.arg()

        self._params['wav_sample_rate'] = int(self._params['wav_sample_rate'])
        self._params['srt_delay_size'] = int(
            self._params['wav_sample_rate'] *
            float(self._params['srt_delay_size_s']))
        self._params['wav_window_size'] = int(
            self._params['wav_sample_rate'] *
            float(self._params['wav_window_size_s']))
        self._params['wav_window_step'] = int(
            self._params['wav_sample_rate'] *
            float(self._params['wav_window_step_s']))

        # sanity check, we want at least one layer.
        self._params['rnn_layers'] = int(self._params.get('rnn_layers', 1))

        self._session_name = param['session_name']
        self._model_name = param['model_name']
        self._epoch_size = param['epoch_size']
        self._epoch_count = param['epoch_count']
        self._max_steps = param['max_steps']
        self._batch_size = param['batch_size']
        self._optimizer = param['optimizer']
        self._learning_rate = param['learning_rate']
        self._regularization_lambda = param['regularization_lambda']

        self._head_hidden_layers_bias = param['head_hidden_layers_bias']
        self._tail_hidden_layers_bias = param['tail_hidden_layers_bias']

        self._nn_hidden_layer_before_rnn = param['head_hidden_layers']
        self._nn_hidden_layer_after_rnn = param['tail_hidden_layers']

        self._checkpoint_source_path = './checkpoints/{}/'.format(
            self._session_name)
        self._checkpoint_target_path = './checkpoints/{}/model.ckpt'.format(
            self._session_name)
        self._tensorboard_log_root = './tensorboards/'
        self._tensorboard_log_path = './tensorboards/{}/'.format(
            self._session_name)

        Parameters.make_dir(self._checkpoint_source_path)
        Parameters.make_dir(self._tensorboard_log_root)

        path_home = os.path.expanduser('~')

        self._dir_cue_training = os.path.join(
            path_home, 'data/vad/{}/'.format(self._params['wav_sample_rate']))
        self._dir_cue_test = os.path.join(
            path_home,
            'data/vad/{}_test/'.format(self._params['wav_sample_rate']))

        if not os.path.isdir(self._dir_cue_training):
            raise Exception('need training dir')
        if not os.path.isdir(self._dir_cue_test):
            raise Exception('need test dir')

        self.load_std_mean(param)

    def load_std_mean(self, param):
        """
        """
        path_home = os.path.expanduser('~')

        name_std_mean = 'mean_std_{}_{}_{}_{}.npz'.format(
            param['wav_feature'],
            param['wav_window_size'],
            param['wav_window_step'],
            param['wav_cepstrum_size'])

        path_std_mean = os.path.join(
            path_home,
            'data/vad/{}/{}'.format(param['wav_sample_rate'], name_std_mean))

        if os.path.isfile(path_std_mean):
            std_mean = np.load(path_std_mean)

            self._params['wav_feature_mean'] = std_mean['mean']
            self._params['wav_feature_std'] = std_mean['std']
        else:
            feature_size = param['wav_cepstrum_size']

            self._params['wav_feature_mean'] = np.zeros((feature_size))
            self._params['wav_feature_std'] = np.ones((feature_size))

        print 'wav_feature_mean: {}'.format(self._params['wav_feature_mean'])
        print 'wav_feature_std: {}'.format(self._params['wav_feature_std'])

    def get_movie_path(self):
        """
        """
        return self._params['movie_path']

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

    def get_max_steps(self):
        """
        """
        return self._max_steps

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

    def get_regularization_lambda(self):
        """
        """
        return self._regularization_lambda

    def get_rnn_cell(self):
        """
        """
        if self._params['rnn_cell'] == 'basiclstm':
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self._params['rnn_unit_num'],
                forget_bias=self._params['lstm_forget_bias'],
                state_is_tuple=True)
        elif self._params['rnn_cell'] == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(
                self._params['rnn_unit_num'],
                use_peepholes=self._params['lstm_use_peephole'],
                forget_bias=self._params['lstm_forget_bias'],
                state_is_tuple=True)

        if self._params['rnn_layers'] > 1:
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                [rnn_cell] * self._params['rnn_layers'], state_is_tuple=True)

        return rnn_cell

    def get_rnn_unit_num(self):
        """
        """
        return self._params['rnn_unit_num']

    def get_rnn_sequence_length(self):
        """
        """
        return self._params['rnn_sequence_length']

    def get_wav_sample_rate(self):
        """
        """
        return self._params['wav_sample_rate']

    def get_wav_feature_type(self):
        """
        """
        return self._params['wav_feature']

    def get_wav_cepstrum_size(self):
        """
        """
        return self._params['wav_cepstrum_size']

    def get_wav_window_size(self):
        """
        """
        return self._params['wav_window_size']

    def get_wav_window_step(self):
        """
        """
        return self._params['wav_window_step']

    def get_wav_feature_mean(self):
        """
        """
        wav_cepstrum_size = self._params['wav_cepstrum_size']

        return self._params.get(
            'wav_feature_mean', np.zeros((wav_cepstrum_size)))

    def get_wav_feature_std(self):
        """
        """
        wav_cepstrum_size = self._params['wav_cepstrum_size']

        return self._params.get(
            'wav_feature_std', np.ones((wav_cepstrum_size)))

    def get_srt_delay_size(self):
        """
        """
        return self._params['srt_delay_size']

    def get_activation_before_rnn(self):
        """
        """
        name = self._params['head_hidden_layers_activation']

        if name == 'tanh':
            return tf.tanh
        elif name == 'lrelu':
            return Parameters.leaky_relu
        else:
            return tf.nn.relu

    def get_activation_after_rnn(self):
        """
        """
        name = self._params['tail_hidden_layers_activation']

        if name == 'tanh':
            return tf.tanh
        elif name == 'lrelu':
            return Parameters.leaky_relu
        else:
            return tf.nn.relu

    def get_hidden_layer_dim_before_rnn(self):
        """
        """
        return list(self._nn_hidden_layer_before_rnn)

    def get_hidden_layer_dim_after_rnn(self):
        """
        """
        return list(self._nn_hidden_layer_after_rnn)

    def get_dropout_prob_after_rnn(self):
        """
        """
        return float(self._params.get("tail_hidden_layers_dropout_prob", 1.0))

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

    def get_dir_cue_test(self):
        """
        """
        return self._dir_cue_test

    def should_use_adam(self):
        """
        """
        return self._optimizer == 'adam'

    def should_add_bias_before_rnn(self):
        """
        """
        return self._head_hidden_layers_bias

    def should_add_bias_after_rnn(self):
        """
        """
        return self._tail_hidden_layers_bias

    def should_add_residual_before_rnn(self):
        """
        """
        return self._params.get('head_hidden_layers_residual', False)

    def should_add_residual_after_rnn(self):
        """
        """
        return self._params.get('tail_hidden_layers_residual', False)

    def should_dropout_after_rnn(self):
        """
        """
        return self._params.get('tail_hidden_layers_dropout', False)
