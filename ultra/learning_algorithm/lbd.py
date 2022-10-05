"""Training and testing the Lipschitz-Bernoulli-Decoupling for unbiased learning to rank.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import zip

import ultra.utils
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm

class LBD(BaseAlgorithm):


    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build LBD')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,  # Learning rate.
            propensity_learning_rate=-1.0,  # Learning rate.
            max_gradient_norm=5.0,  # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',
            grad_penalty=-1.0,  # grad penelty for Lipschitz Decoupling
            bernoulli=0.0  # Bernoulli parameter for Bernoulli Decoupling
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        print("hparam", self.hparams.to_json())
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)
        self.forward_only = forward_only
        self.propensity_model = None

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)
        self.ranking_scores = None
        self.o_tau = None

        self.output_tuple = self.build_rank_matrix_with_ratio(self.max_candidate_num)
        self.output = self.estimate_output(self.output_tuple)

        print([v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

        if not forward_only:
            # Build model
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            self.smooth_loss = 0

            input_feature_list = self.get_input_feature_list(
                self.docid_inputs[:self.rank_list_size])
            input_feature_tensor = tf.concat(input_feature_list, axis=0)

            # Lipschitz Decoupling step: add the grad penalty to loss
            if self.hparams.grad_penalty >= 0:
                with tf.GradientTape() as tape:
                    tape.watch(input_feature_tensor)
                    _, propensities = self.build_rank_matrix_with_ratio(
                        self.rank_list_size, use_bernoulli_decoupling=False, input_tensor=input_feature_tensor)
                    o_f_grad = tape.gradient(
                        tf.reduce_mean(tf.nn.dropout(propensities[0], 0.1)),
                        input_feature_tensor
                    )
                    o_f_grad_square = tf.reduce_sum(tf.square(o_f_grad))
                    self.scalar(o_f_grad_square, "propensity_grad")
                    self.smooth_loss = o_f_grad_square * self.hparams.grad_penalty
            
            # Bernoulli Decoupling step
            relevance_results, propensity_results = self.build_rank_matrix_with_ratio(
                self.rank_list_size, use_bernoulli_decoupling=True)  # [B, T, 2S + 1]

            # Build the loss
            self.supervise_c_loss = self.build_click_loss(relevance_results, propensity_results)
            self.loss = self.supervise_c_loss + self.smooth_loss

            self.scalar(self.supervise_c_loss, "supervise_loss")

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            self.build_update(self.loss)

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def evaluate(self, output):
        # output: (B, T)
        # label: (B, T)
        label = self.labels
        metric_value = None
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, output)
        reshaped_labels = tf.transpose(tf.convert_to_tensor(label))
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                item_name = '%s_%d' % (metric, topn)
                tf.summary.scalar(item_name, metric_value, collections=['eval'])
        return metric_value

    def estimate_output(self, output_tuple):

        relevances, _ = output_tuple
        relevance = relevances[0]  # (B, T)
        self.evaluate(relevance)

        return relevance

    def build_update(self, loss):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if self.hparams.l2_loss > 0:
            for p in params:
                loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

        gradients = tf.gradients(loss, params)
        if self.hparams.max_gradient_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.max_gradient_norm)
        self.norm = tf.global_norm(gradients)

        optimizer = self.optimizer_func(self.learning_rate)
        self.update = optimizer.apply_gradients(zip(gradients, params),
                                                global_step=self.global_step)

    def build_rank_matrix_with_ratio(self, T, use_bernoulli_decoupling=True, **kwargs):
        # Return (B, T, 2S + 1)
        S = self.exp_settings['selection_bias_cutoff']
        if self.ranking_scores is not None and T == S:
            x = self.ranking_scores
        else:
            x = self.get_ranking_scores(self.docid_inputs[:T], self.is_training,
                                                     **kwargs) # (T, B, S_max)
            x = tf.stack(x)  # (T, B, xxx)
            x = tf.transpose(x, (1, 0, 2))  # (B, T, S)
            if T == S:
                self.ranking_scores = x
        relevance_results = []
        propensity_results = []

        relevance_score = x[:, :, 0]  # (B, T)
        propensity_score = x[:, :, 1:(S + 1)]  # (B, T, S)

        if use_bernoulli_decoupling:
            bernoulli_probs = tf.ones_like(propensity_score) * (1 - self.hparams.bernoulli)
            sample = get_bernoulli_sample(bernoulli_probs)
            propensity_score = propensity_score * sample

        relevance_results.append(relevance_score)
        propensity_results.append(propensity_score)
        return relevance_results, propensity_results

    def scalar(self, tensor, name, eval_only=False, train_only=False):
        collections = ['eval'] if eval_only else ['eval', 'train']
        if train_only:
            collections = ['train']
        tf.summary.scalar(
            name,
            tf.reduce_mean(tensor),
            collections=collections)

    def build_click_loss(self, relevance_scores, propensity_scores):
        loss = 0
        T = self.exp_settings['selection_bias_cutoff']
        trained_labels = self.labels[:T]  # (T, B)
        trained_labels = tf.transpose(tf.stack(trained_labels))  # (B, T)
        for i in range(len(relevance_scores)):
            relevance_score = relevance_scores[i]
            propensity_score = propensity_scores[i]

            factual_propensity = []  # (T, B)
            for i in range(T):
                factual_propensity.append(propensity_score[:, i, i])
            factual_propensity = tf.transpose(tf.stack(factual_propensity))  # (B, T)

            loss += self.softmax_loss(factual_propensity + relevance_score, trained_labels)

        return loss

    def step(self, session, input_feed, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: (tf.Session) tensorflow session to use.
            input_feed: (dictionary) A dictionary containing all the input feed data.
            forward_only: whether to do the backward step (False) or only forward (True).

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [self.update,  # Update Op that does SGD.
                           self.loss,  # Loss for this batch.
                           self.train_summary  # Summarize statistics.
                           ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output  # Model outputs
            ]
            if self.ranking_scores is not None:
                output_feed.append(self.ranking_scores)

        outputs = session.run(output_feed, input_feed)

        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[2]
        else:
            return outputs[-1], outputs[1], outputs[0]  # no loss, outputs, summary.

def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return tf.ceil(probs - tf.random_uniform(tf.shape(probs)))