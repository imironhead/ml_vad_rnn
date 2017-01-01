"""
"""
import numpy as np
import tensorflow as tf


class VadModel(object):
    """
    """
    def __init__(self, params):
        """
        """
        self._session_name = params.get_session_name()

        self._checkpoint_source_path = tf.train.latest_checkpoint(
            params.get_checkpoint_source_path())
        self._checkpoint_target_path = \
            params.get_checkpoint_target_path()

        # number of samples of a sequence for training
        self._training_sequence_size = params.get_rnn_sequence_length()

        # wav feature dimension
        self._wav_cepstrum_size = params.get_wav_cepstrum_size()

        # how many samples to delay
        self._srt_delay_size = params.get_srt_delay_size()

        # batch sample weights for delay
        self._batch_sample_weights = tf.placeholder(
            tf.float32,
            [self._training_sequence_size])

        # source data, wav features batch
        self._source_data = tf.placeholder(
            tf.float32,
            [None, self._training_sequence_size, self._wav_cepstrum_size])

        # label data, srt features batch
        self._target_data = tf.placeholder(
            tf.int32, [None, self._training_sequence_size])

        #
        source, last_size = self.build_nn_before_rnn(params, self._source_data)

        # split source to feature list
        source = tf.reshape(
            source, [-1, self._training_sequence_size * last_size])

        source = tf.split(1, self._training_sequence_size, source)

        # batch size
        batch_size = tf.shape(self._source_data)[0]

        # RNN cell factory
        rnn_cell = params.get_rnn_cell()

        self._state = rnn_cell.zero_state(batch_size, tf.float32)

        # build rnn
        outputs, self._last_state = tf.nn.seq2seq.rnn_decoder(
            source, self._state, rnn_cell)

        #
        logits = self.build_nn_after_rnn(params, outputs)

        self._masks = tf.argmax(logits, 1)

        # final
        probabilities = tf.nn.softmax(logits)

        # weights to filter delay
        wgts = tf.tile(self._batch_sample_weights, [batch_size])

        # rnn loss
        total_loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits], [tf.reshape(self._target_data, [-1])], [wgts])

        # regularization losses

        # cost
        total_size = tf.reduce_sum(wgts)

        self._loss = tf.reduce_sum(total_loss) / total_size

        # global step
        initializer_z = tf.constant_initializer(0.0)

        self._global_step = tf.get_variable(
            "gstep", [], trainable=False, initializer=initializer_z)

        # trainer
        self._trainer = self.build_optimizer(params)
        self._trainer = self._trainer.minimize(
            self._loss, global_step=self._global_step)

        # correctness
        correctness = tf.equal(tf.cast(
            tf.argmax(probabilities, 1), tf.int32),
            tf.reshape(self._target_data, [-1]))
        correctness = tf.cast(correctness, tf.float32)
        correctness = tf.mul(correctness, wgts)

        self._judge = tf.reduce_sum(correctness) / total_size

        #
        self._session = tf.Session()

        # restore check point
        if self._checkpoint_source_path is not None:
            tf.train.Saver().restore(
                self._session, self._checkpoint_source_path)
        else:
            self._session.run(tf.global_variables_initializer())

        tf.summary.scalar('training loss', self._loss)
        tf.summary.scalar('training accuracy', self._judge)

        self._summaries = tf.summary.merge_all()
        self._reporter = tf.summary.FileWriter(
            params.get_tensorboard_log_path(), self._session.graph)

    def build_nn_before_rnn(self, params, source):
        """
        """
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.constant_initializer(1.0)
        regularizer_w = tf.contrib.layers.l2_regularizer(
            params.get_regularization_lambda())

        dims = params.get_hidden_layer_dim_before_rnn()
        size = self._wav_cepstrum_size

        if len(dims) > 0:
            source = tf.reshape(source, [-1, self._wav_cepstrum_size])

        for idx, dim in enumerate(dims):
            w = tf.get_variable(
                'bw{}'.format(idx),
                [size, dim],
                initializer=initializer_w,
                regularizer=regularizer_w)

            source = tf.matmul(source, w)

            if params.should_add_bias_before_rnn():
                b = tf.get_variable(
                    'bb{}'.format(idx), [dim], initializer=initializer_b)

                source = source + b

            if params.should_use_relu_before_rnn() and idx + 1 < len(dims):
                source = tf.nn.relu(source)

            size = dim

        return source, size

    def build_nn_after_rnn(self, params, source):
        """
        """
        initializer_w = tf.contrib.layers.xavier_initializer()
        initializer_b = tf.constant_initializer(1.0)
        regularizer_w = tf.contrib.layers.l2_regularizer(
            params.get_regularization_lambda())

        dims = params.get_hidden_layer_dim_after_rnn()
        size = params.get_rnn_unit_num()

        source = tf.concat(1, source)
        source = tf.reshape(source, [-1, params.get_rnn_unit_num()])

        if len(dims) == 0 or dims[-1] != 2:
            dims.append(2)

        for idx, dim in enumerate(dims):
            w = tf.get_variable(
                'aw{}'.format(idx),
                [size, dim],
                initializer=initializer_w,
                regularizer=regularizer_w)

            source = tf.matmul(source, w)

            if params.should_add_bias_after_rnn():
                b = tf.get_variable(
                    'ab{}'.format(idx), [dim], initializer=initializer_b)

                source = source + b

            if params.should_use_relu_after_rnn() and idx + 1 < len(dims):
                source = tf.nn.relu(source)

            size = dim

        return source

    def build_optimizer(self, params):
        """
        """
        if params.should_use_adam():
            optimizer = tf.train.AdamOptimizer(params.get_learning_rate())
        else:
            raise Exception('need specific optimizer')

        return optimizer

    def save_checkpoint(self):
        """
        """
        saver = tf.train.Saver()

        saver.save(self._session, self._checkpoint_target_path,
                   global_step=self._global_step)

    def save_summary(self, gstep, tag=None, value=None, summary=None):
        """
        """
        if (tag is None or value is None) and (summary is None):
            raise Exception('need customized of specified summary')

        if summary is None:
            summary_value = [tf.Summary.Value(tag=tag, simple_value=value)]

            summary = tf.Summary(value=summary_value)

        self._reporter.add_summary(summary, gstep)

    def initial_states(self, source_wav):
        """
        """
        fetch = self._state

        feeds = {
            self._source_data: source_wav[:, :self._training_sequence_size]
        }

        return self._session.run(fetch, feeds)

    def reshape_data(self, source, uni_length=None):
        """
        source:
            the out-most container may be a list
            shape: [batch_size, ?, feature_size]
        uni_length:
            desired length. find minimum length if it's None.

        return:
            shape: [batch_size, k * sequence_size, feature_size]
        """
        if uni_length is None:
            sequence_size = self._training_sequence_size

            min_length = min([r.shape[0] for r in source])

            uni_length = (min_length / sequence_size) * sequence_size

        return np.vstack([r[None, :uni_length] for r in source])

    def work(self, task, states, source_wav, target_srt=None, sample_wgt=None):
        """
        task:
            'train', 'test' or 'detect'.
        states:
            initial states for this batch in this step.
            shape: [batch_size, sequence_size]
        source_wav:
            wav data input.
            shape: [batch_size, sequence_size, feature_size]
        target_srt:
            voice activities mask. for training and testing.
            shape: [batch_size, sequence_size]
        sample_wgt:
            time delay mask. for training and testing. all batch data must has
            same delay mask.
            shape: [sequence_size]

        return:
            'train':  last_state, gstep, summary, loss, correctness, trainer
            'test':   last_state, gstep, summary, loss, correctness
            'detect': last_state, masks
        """
        fetch = [self._last_state]

        feeds = {
            self._state: states,
            self._source_data: source_wav,
        }

        # fetch
        if task == 'train' or task == 'test':
            fetch.extend(
                [self._global_step, self._summaries, self._loss, self._judge])

        if task == 'train':
            fetch.append(self._trainer)

        if task == 'detect':
            fetch.append(self._masks)

        # feeds
        if task == 'train' or task == 'test':
            feeds[self._target_data] = target_srt
            feeds[self._batch_sample_weights] = sample_wgt

        return self._session.run(fetch, feeds)

    def train(self, source_wav, target_srt):
        """
        """
        if self._srt_delay_size > 0:
            paddings = np.zeros((self._srt_delay_size))
            target_srt = [np.hstack((paddings, r)) for r in target_srt]

        source_wav = self.reshape_data(source_wav)
        target_srt = self.reshape_data(target_srt, source_wav.shape[1])

        sequence_size = self._training_sequence_size
        total_size = source_wav.shape[1]
        last_states = self.initial_states(source_wav)

        for base in xrange(0, total_size, sequence_size):
            # REVIEW: do we really need the weights for delays?
            if base <= sequence_size:
                sample_wgt = np.ones((sequence_size))

            if base == 0:
                sample_wgt[:self._srt_delay_size] = 0.0

            result = self.work(
                'train',
                last_states,
                source_wav[:, base:base+sequence_size],
                target_srt[:, base:base+sequence_size],
                sample_wgt)

            last_states = result[0]

        # last_state, gstep, summary, loss, correctness, trainer
        return result

    def test(self, source_wav, target_srt):
        """
        """
        if self._srt_delay_size > 0:
            paddings = np.zeros((self._srt_delay_size))
            target_srt = [np.hstack((paddings, r)) for r in target_srt]

        source_wav = self.reshape_data(source_wav)
        target_srt = self.reshape_data(target_srt, source_wav.shape[1])

        sequence_size = self._training_sequence_size
        total_size = source_wav.shape[1]
        last_states = self.initial_states(source_wav)

        correctness_count = 0.0
        correctness_value = 0.0

        for base in xrange(0, total_size, sequence_size):
            # REVIEW: do we really need the weights for delays?
            if base <= sequence_size:
                sample_wgt = np.ones((sequence_size))

            if base == 0:
                sample_wgt[:self._srt_delay_size] = 0.0

            result = self.work(
                'test',
                last_states,
                source_wav[:, base:base+sequence_size],
                target_srt[:, base:base+sequence_size],
                sample_wgt)

            last_states = result[0]

            correctness_count += result[4].size
            correctness_value += result[4].sum()

        # gstep, correctness
        return result[1], (correctness_value / correctness_count)

    def detect(self, source_wav):
        """
        """
        # REVIEW: generator???
        source_wav = self.reshape_data(source_wav)
        sequence_size = self._training_sequence_size
        total_size = source_wav.shape[1]
        last_states = self.initial_states(source_wav)

        results = []

        for base in xrange(0, total_size, sequence_size):
            result = self.work(
                'detect',
                last_states,
                source_wav[:, base:base+sequence_size])

            last_states = result[0]

            results.append(result[1])

        results = np.hstack(results)

        return results[:, self._srt_delay_size:]
