import tensorflow as tf
import numpy as np


class MRI:
    
    @tf.function 
    def __init__(self, name='MRI', train=True, im_size=156, label_size=3, device=None):
        self.name = name
        self.im_size_squared = im_size**2
        self.label_size = self.output_size = label_size
        self.optimizer = tf.keras.optimizers.Adam(0.00001)
        self.device = device
        if train:
            self.dropout = 0.95
        else:
            self.dropout = 1.0
            
        self.device = None
        
        n_in = self.im_size_squared
        shape1 = tf.constant([n_in, 512])
        self.w_fc1 = tf.Variable(
            tf.random.truncated_normal(shape=shape1, dtype=tf.float32, stddev=0.1), name='w1_{0}'.format(name))
        self.b_fc1 =  tf.Variable(tf.constant(0.0, shape=shape1[-1:], dtype=tf.float32), name='b1_{0}'.format(name))
        
        shape2 = tf.constant([512, 128])
        self.w_fc2 = tf.Variable(
            tf.random.truncated_normal(shape=shape2, dtype=tf.float32, stddev=0.1), name='w2_{0}'.format(name))
        self.b_fc2 =  tf.Variable(tf.constant(0.0, shape=shape2[-1:], dtype=tf.float32), name='b2_{0}'.format(name))

        shape3 = tf.constant([128, self.output_size])
        self.w_fc3 = tf.Variable(
            tf.random.truncated_normal(shape=shape3, dtype=tf.float32, stddev=0.1), name='w3_{0}'.format(name))
        self.b_fc3 =  tf.Variable(tf.constant(0.0, shape=shape3[-1:], dtype=tf.float32), name='b3_{0}'.format(name))
        
        self.variables = [self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2, self.w_fc3, self.b_fc3]
    
    
    @tf.function 
    def forward(self, X):
        """
        Method to do forward pass
        X: Tensor, inputs
        """
        X = tf.reshape(X, [tf.shape(X)[0], -1])
        if self.device is not None:
            with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            # Leave choice of device to default
            self.y = self.compute_output(X)
        return self.y
    
    
    @tf.function
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """

        # Compute values in hidden layer
        self.fc1 = self.fclayer(X, self.w_fc1, self.b_fc1, 'fc1')
        self.fc2 = self.fclayer(self.fc1, self.w_fc2, self.b_fc2, 'fc2')
        dropout2 = tf.nn.dropout(self.fc2, rate=1 - (self.dropout), name='dropout2')

        logits = tf.nn.bias_add(tf.matmul(dropout2, self.w_fc3), self.b_fc3, name='logits')
        self.preds = tf.nn.softmax(logits, name='output')
        return logits
    
    
    @tf.function
    def fclayer(self, x, w_fc, b_fc, name, prop=True):
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w_fc), b_fc), name=name)
        return fc

    
    @tf.function   
    def cost(self, y_pred, y_true):
        return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

    
    @tf.function
    def backward(self, x_batch, y_batch):
        x_batch = tf.reshape(x_batch, [tf.shape(x_batch)[0], -1])
        with tf.GradientTape() as tape:
            predicted = self.forward(x_batch)
            current_loss = self.cost(predicted, y_batch)
        grads = tape.gradient(current_loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables))
        return None

        
##########################################################################################################        
        
        
class MRI_2dCNN:
    
    #@tf.function        
    def __init__(self, name='MRI_2dCNN', train=True, im_size=156, label_size=3, device=None):
        
        self.name = name
        self.im_size = tf.constant(im_size, dtype=tf.float32)
        self.im_size_squared = im_size**2
        self.label_size = self.output_size = label_size
        self.optimizer = tf.keras.optimizers.Adam(0.00001)
        self.device = device       
        if train:
            self.dropout = tf.constant(0.95, dtype=tf.float32)
        else:
            self.dropout = tf.constant(1.0, dtype=tf.float32)
            
        shape1 = tf.constant([3, 3, 1, 32])
        self.w1_conv = tf.Variable(tf.random.truncated_normal(
            shape=shape1, dtype=tf.float32, stddev=0.1), name='w1_{0}'.format(name))
        self.b1_conv = tf.Variable(tf.constant(0.0, shape=shape1[-1:], dtype=tf.float32), name='b1_{0}'.format(name))
        
        shape2 = tf.constant([3, 3, 32, 64])
        self. w2_conv = tf.Variable(tf.random.truncated_normal(
            shape=shape2, dtype=tf.float32, stddev=0.1), name='w2_{0}'.format(name))
        self.b2_conv = tf.Variable(tf.constant(0.0, shape=shape2[-1:], dtype=tf.float32), name='b2_{0}'.format(name))
        
        shape_flat = tf.constant(int(shape2[-1] * self.im_size_squared / 4)) #division due to pooling
        shape3 = tf.constant([int(shape_flat), 512])
        self.w_fc1 = tf.Variable(
            tf.random.truncated_normal(shape=shape3, dtype=tf.float32, stddev=0.1), name='w3_{0}'.format(name))
        self.b_fc1 =  tf.Variable(tf.constant(0.0, shape=shape3[-1:], dtype=tf.float32), name='b3_{0}'.format(name))
        
        shape4 = tf.constant([512, self.output_size])
        self.w_fc2 = tf.Variable(
            tf.random.truncated_normal(shape=shape4, dtype=tf.float32, stddev=0.1), name='w4_{0}'.format(name))
        self.b_fc2 =  tf.Variable(tf.constant(0.0, shape=shape4[-1:], dtype=tf.float32), name='b4_{0}'.format(name))

        self.variables = [self.w1_conv, self.b1_conv, self.w2_conv, self.b2_conv, self.w_fc1, self.b_fc1, 
                          self.w_fc2, self.b_fc2]
    
    @tf.function    
    def forward(self, X):
        """
        Method to do forward pass
        X: Tensor, inputs
        """
        
        if tf.shape(tf.shape(X)) == 3:
            X = tf.expand_dims(X, 3)
        if self.device is not None:
            with tf.device('gpu:1' if self.device=='gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            # Leave choice of device to default
            self.y = self.compute_output(X)
        return self.y
   

    @tf.function
    def convlayer(self, x, w_conv, b_conv, name):
        conv = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=x, filters=w_conv, strides=[1, 1, 1, 1], padding='SAME'), b_conv), name=name)
        return conv
    
    
    @tf.function
    def fclayer(self, x, w_fc, b_fc, name, prop=True):
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w_fc), b_fc), name=name)
        return fc
    
    
    @tf.function
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        
        # Compute values in hidden layer
        self.conv1 = self.convlayer(X, self.w1_conv, self.b1_conv, 'conv1')
        self.conv2 = self.convlayer(self.conv1, self.w2_conv, self.b2_conv, 'conv2')
        #try different strides
        max_pool1 = tf.nn.max_pool2d(input=self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool1')
        flatten = tf.reshape(max_pool1, [tf.shape(max_pool1)[0], -1])
        self.fc1 = self.fclayer(flatten, self.w_fc1, self.b_fc1, 'fc1')
        
        dropout = tf.nn.dropout(self.fc1, rate=1 - (self.dropout), name='dropout2')
        logits = tf.nn.bias_add(tf.matmul(dropout, self.w_fc2), self.b_fc2, name='logits')
        self.preds = tf.nn.softmax(logits, name='output')
        return logits
    
    
    @tf.function   
    def cost(self, y_pred, y_true):
        return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    
    
    @tf.function
    def backward(self, x_batch, y_batch):
        if tf.shape(tf.shape(x_batch)) == 3:
            x_batch = tf.expand_dims(x_batch, 3)
        with tf.GradientTape() as tape:
            predicted = self.forward(x_batch)
            current_loss = self.cost(predicted, y_batch)
        grads = tape.gradient(current_loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables))
