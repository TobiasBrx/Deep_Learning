import tensorflow as tf
import numpy as np

im_size = 156
im_size_squared = im_size**2
label_size = 5
output_size=5


class MRI:
  
    def __init__(self, name='MRI', train=True):
        self.name = name
        if train:
            self.dropout = 0.95
        else:
            self.dropout = 1.0
            
        #self.labels = tf.reshape(labels, [-1, label_size], name="input_label")
        #activations += [images, ]
        
        #self.flatten = tf.reshape(images, [batch_size, -1])
        #activations += [flatten, ]
        
        self.device = None
        
        n_in = im_size_squared
        shape1 = [n_in, 8]
        self.w_fc1 = tf.Variable(
            tf.random.truncated_normal(shape=shape1, dtype=tf.float32, stddev=0.1), name='w1_{0}'.format(name))
        self.b_fc1 =  tf.Variable(tf.constant(0.0, shape=shape1[-1:], dtype=tf.float32), name='b1_{0}'.format(name))
        
            #activations += [fc1, ]

#         with tf.variable_scope('conv1'):
#             w_conv1, b_conv1, conv1 = self.convlayer(images, [3, 3, 1, 32], 'conv1')
#             activations += [conv1, ]

#         with tf.variable_scope('conv2'):
#             w_conv2, b_conv2, conv2 = self.convlayer(conv1, [3, 3, 32, 64], 'conv2')
#             activations += [conv2, ]

#         with tf.variable_scope('max_pool1'):
#             max_pool1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool1')
#             activations += [max_pool1, ]

#         with tf.variable_scope('flatten'):
#             flatten = tf.contrib.layers.flatten(max_pool1)
#             activations += [flatten, ]
        shape2 = [8, 5]
        self.w_fc2 = tf.Variable(
            tf.random.truncated_normal(shape=shape2, dtype=tf.float32, stddev=0.1), name='w2_{0}'.format(name))
        self.b_fc2 =  tf.Variable(tf.constant(0.0, shape=shape2[-1:], dtype=tf.float32), name='b2_{0}'.format(name))
        #activations += [fc2, ]

        shape3 = [5, output_size]
        self.w_fc3 = tf.Variable(
            tf.random.truncated_normal(shape=shape3, dtype=tf.float32, stddev=0.1), name='w3_{0}'.format(name))
        self.b_fc3 =  tf.Variable(tf.constant(0.0, shape=shape3[-1:], dtype=tf.float32), name='b3_{0}'.format(name))
        
        self.variables = [self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2, self.w_fc3, self.b_fc3]
            
    def forward(self, X):
        """
        Method to do forward pass
        X: Tensor, inputs
        """
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
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # Compute values in hidden layer
        self.fc1 = self.fclayer(X, self.w_fc1, self.b_fc1, 'fc1') # was 512
        self.fc2 = self.fclayer(self.fc1, self.w_fc2, self.b_fc2, 'fc2') # was 512
        dropout2 = tf.nn.dropout(self.fc2, rate=1 - (self.dropout), name='dropout2')

        logits = tf.nn.bias_add(tf.matmul(dropout2, self.w_fc3), self.b_fc3, name='logits')
        self.preds = tf.nn.softmax(logits, name='output')
        self.variables = [self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2, self.w_fc3, self.b_fc3]
        return logits # ,activations

    
    def convlayer(self, input, shape, name):
        w_conv = tf.Variable(tf.random.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
        b_conv = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
        conv = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=input, filters=w_conv, strides=[1, 1, 1, 1], padding='SAME'), b_conv), name=name)
        return w_conv, b_conv, conv
    
    @tf.function
    def fclayer(self, x, w_fc, b_fc, name, prop=True):
        #w_fc = tf.Variable(tf.random.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
        #b_fc = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w_fc), b_fc), name=name)
        return fc

    @tf.function   
    def cost(self, y_pred, y_true):
        return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

    def predict(self, images, labels, batch_size, reuse=False, train=True):

        
        #with tf.compat.v1.variable_scope(self.name):
        
        #    if reuse:
        #        scope.reuse_variables()
        

            #activations = []
               
            #images = tf.reshape(images, [-1, im_size, im_size], name="input")
            #activations += [images, ]
        return
            

    @property
    def params(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def backward(self, x_batch, y_batch):
        optimizer = tf.keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            predicted = self.forward(x_batch)
            current_loss = self.cost(predicted, y_batch)
        grads = tape.gradient(current_loss, self.variables)
        first_var = min(self.variables, key=lambda x: x.name)
        optimizer.apply_gradients(zip(grads, self.variables))
        #optimizer._distributed_apply(strategy, zip(grads, self.variables))

class MRI_2dCNN:
  
    def __init__(self, name='MRI_2dCNN', train=True):
        self.name = name
        if train:
            self.dropout = 0.5
        else:
            self.dropout = 1.0

    def convlayer(self, input, shape, name):
        w_conv = tf.Variable(tf.random.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
        b_conv = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input=input, filters=w_conv, strides=[1, 1, 1, 1], padding='SAME'), b_conv), name=name)
        return w_conv, b_conv, conv
  
    def fclayer(self, input, shape, name, prop=True):
        w_fc = tf.Variable(tf.random.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
        b_fc = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
        if prop:
            fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w_fc), b_fc), name=name)
            return w_fc, b_fc, fc
        else:
            return w_fc, b_fc

    def __call__(self, images, labels, reuse=False, train=True):
        with tf.compat.v1.variable_scope(self.name):
      
            if reuse:
                scope.reuse_variables()

            activations = []
               
            with tf.compat.v1.variable_scope('input'):
                images = tf.reshape(images, [-1, im_size, im_size,1], name="input")
                activations += [images, ]
                
            with tf.compat.v1.variable_scope('label'):
                labels = tf.reshape(labels, [-1, output_size], name="label")
                activations += [labels, ]
                
            with tf.compat.v1.variable_scope('conv1'):
                w_conv1, b_conv1, conv1 = self.convlayer(images, [3, 3, 1, 32], 'conv1')
                activations += [conv1, ]

            with tf.compat.v1.variable_scope('conv2'):
                w_conv2, b_conv2, conv2 = self.convlayer(conv1, [3, 3, 32, 64], 'conv2')
                activations += [conv2, ]

            with tf.compat.v1.variable_scope('max_pool1'):
                max_pool1 = tf.nn.max_pool2d(input=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool1')
                activations += [max_pool1, ]

            with tf.compat.v1.variable_scope('flatten'):
                flatten = tf.contrib.layers.flatten(max_pool1)
                activations += [flatten, ]

            with tf.compat.v1.variable_scope('fc_1'):
                n_in = int(flatten.get_shape()[1])
                w_fc1, b_fc1, fc1 = self.fclayer(flatten, [n_in, 512], 'fc2') # was 512
                activations += [fc1, ]

            with tf.compat.v1.variable_scope('dropout2'):
                dropout2 = tf.nn.dropout(fc1, rate=1 - (self.dropout), name='dropout2')

            with tf.compat.v1.variable_scope('output'):
                w_fc3, b_fc3 = self.fclayer(dropout2, [512, output_size], 'fc3', prop=False)
                logits = tf.nn.bias_add(tf.matmul(dropout2, w_fc3), b_fc3, name='logits')
                preds = tf.nn.softmax(logits, name='output')
                activations += [preds, ]

                return activations, logits

    @property
    def params(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
