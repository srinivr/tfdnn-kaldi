# Author: "Srinivas Venkattaramanujam"
# Version: = 0.5
# Date: "27/12/2016"
# Copyright 2016  Srinivas Venkattaramanujam (author: Srinivas Venkattaramanujam)
# Licensed under the Apache License, Version 2.0 (the "License")
import tensorflow as tf
import math
class HiddenLayer:
    def __init__(self,input_dim, output_dim, inputs, scope_name):
        with tf.variable_scope(scope_name, reuse=False):
            self.weights = tf.get_variable('weights',(input_dim,output_dim),initializer=tf.truncated_normal_initializer(0.0,1/math.sqrt(input_dim)))
            self.biases = tf.get_variable('biases',(output_dim),initializer=tf.constant_initializer(0.0))
            self.activations = tf.sigmoid(tf.matmul(inputs,self.weights) + self.biases,'activations')

class LinearLayer:
    def __init__(self,input_dim, output_dim, inputs, scope_name):
        with tf.variable_scope(scope_name, reuse=False):
            self.weights = tf.get_variable('weights',(input_dim,output_dim),initializer=tf.truncated_normal_initializer(0.0,1/math.sqrt(input_dim)))
            self.biases = tf.get_variable('biases',(output_dim),initializer=tf.constant_initializer(0.0))
            self.activations = tf.matmul(inputs,self.weights) + self.biases

class RecurrentLayer:
    pass