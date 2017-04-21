#!/usr/bin/env python
# Author: "Srinivas Venkattaramanujam"
# Version: = 0.5
# Date: "27/12/2016"
# Copyright 2016  Srinivas Venkattaramanujam (author: Srinivas Venkattaramanujam)
# Licensed under the Apache License, Version 2.0 (the "License")

import time
import tensorflow as tf
from neuralnetworks.dnn import DNN
from neuralnetworks.trainer import Trainer
from utils.utils import Utils, FeatureUtils
from kaldi_io import kaldi_io
import numpy as np
from utils.config import Config

begin_time = time.time()
TRAIN = True
DISCRIMINATIVE_PRETRAINING = True
TEST = True

# TODO: Seperate learning scheduler from trainer.
# TODO: Add kaldi type scheduler.
# TODO: Make a config file (instead of current config.py)
# TODO: Move path strings to config
# TODO: Add kaldi commands (to apply cmvn, add deltas, splice feats,
#       ali-to-pdf | pdf-to-post | post-to-feat in a file and make the process 'one step to run'


layers = [759, 1024, 1024, 1024, 1024, 1954]
dnn = DNN(layers, None, True)

dnn.buildForwardGraph(256, discrimivative=DISCRIMINATIVE_PRETRAINING)
dnn.buildTrainGraph()
dnn.buildEvalGraph()

featureUtils = FeatureUtils()
utils = Utils()

with tf.Session() as sess:
    # Saver
    saver = tf.train.Saver()
    if TRAIN:
        prior = np.zeros(1954)

        # read train feats and alignments
        tr_feat_keys, tr_feats = featureUtils.read_feats(Config.TRAIN_FEATS)
        tr_ali_keys, tr_alis, prior = featureUtils.read_ali_and_compute_prior(Config.TRAIN_ALIGNMENTS, prior)
        utils.verify_order(tr_feat_keys, tr_ali_keys)

        # read cross-validation feats and alignments
        cv_feat_keys, cv_feats = featureUtils.read_feats(Config.CV_FEATS)
        cv_ali_keys, cv_alis, prior = featureUtils.read_ali_and_compute_prior(Config.CV_ALIGNMENTS, prior)
        utils.verify_order(cv_feat_keys, cv_ali_keys)
        featureUtils.save_prior(prior)
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.initialize_all_variables())
        eval_loss, eval_ce = dnn.eval(sess, cv_feats, cv_alis)
        print 'validation loss before training:', eval_loss, 'xent:', eval_ce

        epochs = 2000
        epoch = 0
        step = 0
        # for epoch in range(epochs):
        lr = 0.1
        trainer = Trainer(dnn, saver, eval_ce, lr)
        if DISCRIMINATIVE_PRETRAINING:
            PRETRAIN_EPOCHS = 6
            print 'Discriminative Layerwise Pretraining...'
            # trainer.train(sess, 1, tr_feats, tr_alis, tr_feat_keys, tr_ali_keys, cv_feats, cv_alis)
            for i in range(1, len(layers) - 1):
                print 'Pre-training Layer ', i
                # sess.run(tf.initialize_variables(dnn.addLayer(i)))
                params = dnn.addLayer(i)
                print 'params are again', params
                sess.run(tf.variables_initializer(params))
                trainer.train(sess, PRETRAIN_EPOCHS, tr_feats, tr_alis, tr_feat_keys, tr_ali_keys, cv_feats, cv_alis)

        print 'fine tuning network...'
        trainer.train(sess, epochs, tr_feats, tr_alis, tr_feat_keys, tr_ali_keys, cv_feats, cv_alis)

    if TEST:
        saver.restore(sess, '/speech1/DIT_PROJ/srini/PycharmProjects/tfkaldi-fork/models/model-200000')
        # sess.run(tf.initialize_all_variables())

        # evaluate to verify if correct model is restored
        cv_feat_keys, cv_feats = featureUtils.read_feats(Config.CV_FEATS)
        cv_ali_keys, cv_alis, _ = featureUtils.read_ali_and_compute_prior(Config.CV_ALIGNMENTS, None)

        utils.verify_order(cv_feat_keys, cv_ali_keys)
        eval_loss, eval_ce = dnn.eval(sess, cv_feats, cv_alis)
        print 'validation loss after model restore:', eval_loss, 'xent:', eval_ce

        print 'Forward pass test data'

        te_key, te_mat, te_feat_len = featureUtils.read_feats(Config.TEST_FEATS, True)
        out = dnn.forwardPass(sess, te_mat)
        out = np.array(out)
        # loading prior
        prior = np.loadtxt('/speech1/DIT_PROJ/srini/PycharmProjects/tfkaldi-fork/prior.npy', dtype=np.float32)
        out /= prior
        np.where(out == 0, np.finfo(float).eps, out)
        out = np.log(out)
        prev = 0
        with open('out2.ark', 'wb') as f:
            for ran in range(len(te_key)):
                temp = out[0][prev:prev + te_feat_len[ran]]
                prev = prev + te_feat_len[ran]
                kaldi_io.write_mat(f, temp, key=te_key[ran])

    end_time = time.time()
    print 'seconds', (end_time - begin_time)
