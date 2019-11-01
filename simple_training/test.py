from tensorflow.python.ops    import gen_nn_ops

import numpy                as np
import tensorflow           as tf
import matplotlib.pyplot    as plt
from utils import _parse_function
from model import MRI, MRI_2dCNN, im_size, im_size_squared, output_size
from train import logdir, chkpt

resultsdir = 'results/'
batch_size = 100

class Tester:
    def __init__(self):
        
        self.epsilon = 1e-10
        with tf.variable_scope('2dCNN_y'):
            self.model = MRI_2dCNN(train=False)
            self.X = tf.placeholder(tf.float32, [None, im_size_squared], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_size], name='y')
            self.activations, self.logits = self.model(self.X)
            #self.l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in self.model.params if 'b' not in p.name]) * 0.001
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))# + self.l2_loss
            self.preds = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))
    
          
    def run(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(logdir))                                
            self.test_filenames = tf.placeholder(tf.string, shape=[None])
            self.test_dataset = tf.data.TFRecordDataset(self.test_filenames).map(_parse_function).shuffle(buffer_size=1000).batch(batch_size).repeat()
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            self.test_next_element = self.test_iterator.get_next()
            self.testing_filenames = ["stesting_flat_y_156_full_dataset.tfrecords"]
            sess.run(self.test_iterator.initializer, feed_dict={self.test_filenames: self.testing_filenames})
            
            self.test(sess)    
    
    def test(self, sess):
        n_batches = 100

        avg_accuracy = 0
        for batch in range(n_batches):
            x_batch, y_batch, _name = sess.run(self.test_next_element)
            avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: x_batch, self.y: y_batch})[0]
        avg_accuracy /= n_batches
        print('Testing Accuracy {0:6.4f}'.format(avg_accuracy))

if __name__ == '__main__':
    Tester().run()