# Author: "Srinivas Venkattaramanujam"
# Version: = 0.5
# Date: "27/12/2016"
# Copyright 2016  Srinivas Venkattaramanujam (author: Srinivas Venkattaramanujam)
# Licensed under the Apache License, Version 2.0 (the "License")
import tensorflow as tf

from layer import HiddenLayer, LinearLayer


class DNN:
    def __init__(self, layer_dims, activations, visualize=False):
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        print 'input dim', self.input_dim, 'output dim', self.output_dim
        self.layer_dims = layer_dims
        self.activations = activations
        self.visualize = visualize
        self.layers = []

    def buildForwardGraph(self, batch_size, discrimivative=False):
        """

        :param batch_size: Minibatch Size. Currently unused. Using None.
        :param discrimivative: True for discriminative pretraining (Creates a graph with zero hidden layers). Default \
        value: False (Creates a graph with specified hidden layers)
        """
        with tf.variable_scope('forward_variables', reuse=False):
            self.input = tf.placeholder(tf.float32, (None, self.input_dim), 'input_nodes')
            self.output = tf.placeholder(tf.float32, (None, self.output_dim), 'output_nodes')
            inpt = self.input;
            if not discrimivative:
                inpt = self.__buildFullGraph__()
                self.layers.append(LinearLayer(self.layer_dims[-2], self.layer_dims[-1], inpt,
                                               str(len(self.layer_dims) - 2) + 'layerNet_output'))
            else:
                self.layers.append(
                    LinearLayer(self.layer_dims[0], self.layer_dims[-1], inpt, '0layerNet_output'))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.step_incr = tf.assign_add(self.global_step, 1)

    def buildTrainGraph(self):
        with tf.variable_scope('train_variables'):
            self.learning_rate = tf.Variable(initial_value=0.1, name='learning_rate', trainable=False)
            self.momentum = tf.Variable(0.9, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=self.momentum)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.__buildLossGraph__()
            if self.visualize == True:
                self.buildSummaryGraph()
            self.learning_rate_half_op = tf.assign(self.learning_rate, self.learning_rate / 2.0)

    def buildEvalGraph(self):
        with tf.variable_scope('eval_variables', reuse=False):
            self.logits = tf.nn.softmax(self.layers[-1].activations, name='logits')
            self.correct_predication = tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predication, tf.float32))

    def buildSummaryGraph(self):
        """
            TODO: Have to fix summary update in case of discriminative pre-training
        """
        # TODO: Have to fix summary update in case of discriminative pre - training
        self.summaryWriter = tf.train.SummaryWriter('logdir5', tf.get_default_graph())
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.scalar_summary(self.learning_rate.op.name, self.learning_rate)
        self.summary = tf.merge_all_summaries()

    def train(self, session, input, output, lr=None, step=0):
        """

        :param session: current TensorFlow session
        :param input: input matrix
        :param output: output matrix in one hot
        :param lr: learning rate. Default value: 0.1
        :param step: global step. TODO: Have to check if this makes sense after the new change
        :return:
        """
        if lr is not None:
            tf.assign(self.learning_rate, lr)

        if self.visualize == True:
            loss, accuracy, _, summary, global_step = session.run(
                [self.loss, self.accuracy, self.train_op, self.summary, self.step_incr],
                feed_dict={self.input: input, self.output: output})
            self.summaryWriter.add_summary(summary, global_step=step)
        else:
            loss, accuracy, _ = session.run([self.loss, self.accuracy, self.train_op],
                                            feed_dict={self.input: input, self.output: output})
        return loss, accuracy

    def addLayer(self, idx):
        """
        :param idx: index of the layer(in the list passed to initialize the network) to be added. Note 0 is the input\
         layers
        :return: return a list of newly created variables that has to initialized
        """
        with tf.variable_scope('forward_variables', reuse=False):
            self.layers = self.layers[:-1]
            print 'layers len', len(self.layers)
            if len(self.layers) == 0:
                inpt = self.input
            else:
                inpt = self.layers[-1].activations
            self.layers.append(HiddenLayer(self.layer_dims[idx - 1], self.layer_dims[idx], inpt, 'layer' + str(idx)))
            self.layers.append(
                LinearLayer(self.layer_dims[-2], self.layer_dims[-1], self.layers[-1].activations,
                            str(idx) + 'layerNet_output'))
            self.__buildLossGraph__()

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward_variables/layer' + str(idx))
        params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward_variables_' + str(idx))
        params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope='forward_variables/' + str(idx) + 'layerNet_output')
        print 'params are ', params
        self.buildEvalGraph()
        self.buildSummaryGraph()
        return params

    def __buildLossGraph__(self):
        self.xent = tf.nn.softmax_cross_entropy_with_logits(self.layers[-1].activations, self.output)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(self.xent, name='loss') + 0.0002 * sum(reg_losses)
        self.train_op = self.optimizer.minimize(self.loss)

    def __buildFullGraph__(self):
        inpt = self.input
        for idx in range(1, len(self.layer_dims) - 1):
            self.layers.append(HiddenLayer(self.layer_dims[idx - 1], self.layer_dims[idx], inpt, 'layer' + str(idx)))
            inpt = self.layers[-1].activations
        return inpt

    def forwardPass(self, sess, inpt):
        return sess.run([self.logits], feed_dict={self.input: inpt})

    def eval(self, session, input, output):
        accuracy, loss = session.run([self.accuracy, self.loss], feed_dict={self.input: input, self.output: output})
        return accuracy, loss

    def reduce_learning_rate(self, sess):
        """

        :param sess: current TensorFlow session
        :return: updated learning rate
        """
        return sess.run(self.learning_rate_half_op)

    def update_momementum(self, sess, mom):
        sess.run(tf.assign(self.momentum, mom))
