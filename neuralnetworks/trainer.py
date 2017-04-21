# Author: "Srinivas Venkattaramanujam"
# Version: = 0.5
# Date: "27/12/2016"
# Copyright 2016  Srinivas Venkattaramanujam (author: Srinivas Venkattaramanujam)
# Licensed under the Apache License, Version 2.0 (the "License")
import math,sys
from utils.utils import Utils
class Trainer:
    def __init__(self, network, saver, initial_validation_xent, initial_lr):
        self.network = network
        self.saver = saver
        self.threshold = 0.001
        self.fix_lr_for_epochs = 2
        self.eval_xent = initial_validation_xent
        self.lr = initial_lr
        self.MB_SIZE = 256
        self.step = 0

    def train(self, sess, epochs, train_inputs, train_outputs, train_input_keys, train_output_keys, cv_feats, cv_outputs):
        epoch = 0
        utils = Utils()
        while epoch < epochs:
            avg_epoch_loss = 0.0
            utils.shuffle_together([train_inputs, train_outputs, train_input_keys, train_output_keys])
            utils.verify_order(train_input_keys, train_output_keys)
            num_iters = int(math.floor(len(train_inputs) / float(self.MB_SIZE)))
            for iter in range(num_iters):
                batch_xs, batch_ys = train_inputs[iter * self.MB_SIZE:(iter + 1) * self.MB_SIZE], train_outputs[
                                                                                  iter * self.MB_SIZE:(iter + 1) * self.MB_SIZE]
                loss, accuracy = self.network.train(sess, batch_xs, batch_ys, None, self.step)
                self.step += 1
                avg_epoch_loss += loss
                print 'loss at epoch', epoch, 'step', iter, ':', loss, 'accuracy:', (accuracy * 100), '%'
                if (iter + 1) % 10 == 0:
                    sys.stdout.flush()
            avg_epoch_loss /= num_iters
            print 'avg epoch loss', avg_epoch_loss
            # evaluating the performance
            eval_loss_, eval_ce_ = self.network.eval(sess, cv_feats, cv_outputs)
            print 'validation loss at epoch', epoch, ':', (eval_loss_ * 100), '%,', eval_ce_
            if not self.schedule(sess, epoch, eval_ce_):
                break
            save_path = self.saver.save(sess, 'models/model', global_step=epoch)
            print 'model stored at', save_path
            sys.stdout.flush()
            epoch += 1
        self.saver.save(sess, 'models/model', global_step=200000) # Figure out a better way!

    def schedule(self, sess, epoch, xent):
        if epoch >= self.fix_lr_for_epochs and self.eval_xent < xent:  # (eval_ce - eval_ce_ <=0.01 or eval_ce_ >= eval_ce):
            epoch -= 1
            self.saver.restore(sess, 'models/model-' + str(epoch))
            lr_ = self.network.reduce_learning_rate(sess)
            while lr_ > self.lr:
                lr_ = self.network.reduce_learning_rate(sess)
            self.lr = lr_
            if self.lr < 0.001:
                'learning rate is less than threshold. Exiting...'  # Doing 5 epochs with fixed lr...'
                return False
            print 'reducing learning rate in epoch and restoring model', (epoch + 1), 'lr:', self.lr
        if epoch >= 2:
            self.eval_xent = xent
        return True