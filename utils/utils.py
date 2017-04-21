# Copyright 2016  Srinivas Venkattaramanujam (author: Srinivas Venkattaramanujam)
# Licensed under the Apache License, Version 2.0 (the "License")
import random
from kaldi_io import kaldi_io
import numpy as np
# TODO: Make the classes static and methods to functions
class Utils:
    def __init__(self):
        pass

    def shuffle_together(self, mats):
        """

        :param mats: shuffles the given matrices and maintains the same 'shuffled order' in all matrices
        """
        rng = random.getstate()
        for mat in mats:
            random.setstate(rng) # reset random state to the saved state to get the same 'shuffled order' as previous shuffling
            random.shuffle(mat)

    def verify_order(self, list1, list2):
        """

        :list1: some list
        :list2: some list
        """
        for l1, l2 in zip(list1, list2):
            assert l1 == l2


class FeatureUtils:
    def __init__(self):
        pass

    def read_feats(self, path, return_feat_len=False):
        """

        :param path: path to features in Kaldi format
        :param return_feat_len: If True return number of features for each utterance. Useful in decoding
        :return: return feature keys, feature matrix [if return_feat_lean is True, returns feat_len]
        """
        mat = []
        keys = []
        feat_len = []
        print 'reading feats:', path
        for _key, _mat in kaldi_io.read_mat_scp(path):
            mat.extend(_mat)
            keys.append(_key)
            if return_feat_len == True:
                feat_len.append(len(_mat))

        if return_feat_len:
            return keys, mat, feat_len
        else:
            return keys, mat

    def read_ali_and_compute_prior(self, path, initial_prior):
        """

        :param path: path to read alignments. Alignments must be in Kaldi feature format with one hot encoding. TODO: Check sparse options
        :param initial_prior: initial priors in same shape. if None, priors are not calculated
        :return: alignment keys, alignements, prior
        """
        tr_ali = []
        key2 = []
        print 'reading train alignments'
        prior = initial_prior
        for _key, _ali in kaldi_io.read_mat_scp(path):
            tr_ali.extend(_ali)
            key2.append(_key)
            if not initial_prior is None:
                prior = np.add(prior, np.asarray(_ali).sum(axis=0))
        return key2, tr_ali, prior

    def save_prior(self, prior):
        """

        :param prior: prior values
        """
        prior += 0.5
        np.savetxt('unnormalized_prior.npy', prior)
        prior = prior / prior.sum()
        np.savetxt('prior.npy', prior)
        print 'Saved priors to prior.npy...'
