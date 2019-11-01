from tensorflow.python.ops    import gen_nn_ops

import numpy                as np
import tensorflow           as tf
import matplotlib.pyplot    as plt
from utils import _parse_function, visualize, heatmap_ayo
from model import MRI,MRI_2dCNN, im_size, im_size_squared, output_size
from train import logdir, chkpt

resultsdir = 'results_x_neg_FF/'

class LayerwiseRelevancePropagation:
    def __init__(self):
        self.epsilon = 1e-10
    #with tf.variable_scope('feed_forward'):
#         self.model = MRI(train=False)
#         self.X = tf.placeholder(tf.float32, [None, im_size_squared], name='X')
#         self.y = tf.placeholder(tf.float32, [None, output_size], name='y')
#         self.activations, self.logits = self.model(self.X)
#         #self.l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in self.model.params if 'b' not in p.name]) * 0.001
#         self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))# + self.l2_loss
#         self.preds = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
#         self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
            saver.restore(sess, tf.train.latest_checkpoint(logdir))    
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FFNN_x')
            self.activations = tf.get_collection('LayerwiseRelevancePropagation') #tf.GraphKeys.GLOBAL_VARIABLES,scope=
        print('activations', self.activations)    
        self.X = self.activations[0]
        self.Y = self.activations[1]
        
        self.act_weights = {}
        for act in self.activations[4:]:
            print('activations', act)
            for wt in weights:
                name = act.name.split('/')[2]
                if name == wt.name.split('/')[2]:
                    if name not in self.act_weights:
                        self.act_weights[name] = wt
                        
        self.activations = self.activations[:3:-1] + [self.activations[2]]
        print('activations reversed', self.activations)    
        self.relevances = self.get_relevances()

    def get_relevances(self):
        print('activations 0', self.activations[0])
        print('label', self.Y)
        relevances = [self.activations[0] * self.Y, ]
        for i in range(1, len(self.activations)):
            name = self.activations[i - 1].name.split('/')[2]
            if 'output' in name or 'fc' in name:
                relevances.append(self.backprop_fc(name, self.activations[i], relevances[-1]))
            elif 'flatten' in name:
                relevances.append(self.backprop_flatten(self.activations[i], relevances[-1]))
            elif 'max_pool' in name:
                relevances.append(self.backprop_max_pool2d(self.activations[i], relevances[-1]))
            elif 'conv' in name:
                relevances.append(self.backprop_conv2d(name, self.activations[i], relevances[-1]))
            else:
                raise 'Error parsing layer!'   
        print('relevances', relevances)
        return relevances

#LRP FUNCTIONS WITH NEGATIVE RELEVANCE
    def backprop_fc(self, name, activation, relevance):
        alpha = 2
        beta = 1
        w = self.act_weights[name]
        w_pos = tf.maximum(0.0, w)
        z = tf.matmul(activation, w_pos) + self.epsilon
        s = relevance / z
        c = tf.matmul(s, tf.transpose(w_pos))
        w_neg = tf.maximum(0.0, -w)
        z_ = tf.matmul(activation, w_neg) + self.epsilon
        s_ = relevance / z_
        c_ = tf.matmul(s_, tf.transpose(w_neg))
        return (alpha* c - beta * c_) * activation
     
    def backprop_flatten(self, activation, relevance):
        shape = activation.get_shape().as_list()
        shape[0] = -1
        return tf.reshape(relevance, shape)

    def backprop_max_pool2d(self, activation, relevance, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        z = tf.nn.max_pool(activation, ksize, strides, padding='SAME') + self.epsilon
        s = relevance / z
        c = gen_nn_ops.max_pool_grad_v2(activation, z, s, ksize, strides, padding='SAME')
        return c * activation

    def backprop_conv2d(self, name, activation, relevance, strides=[1, 1, 1, 1]):
        alpha = 2
        beta = 1
        w = self.act_weights[name]
        w_pos = tf.maximum(0.0, w)
        z = tf.nn.conv2d(activation, w_pos, strides, padding='SAME') + self.epsilon
        s = relevance / z
        c = tf.nn.conv2d_backprop_input(tf.shape(activation), w_pos, s, strides, padding='SAME')
        w_neg = tf.maximum(0.0, -w)
        z_ = tf.nn.conv2d(activation, w_neg, strides, padding='SAME') + self.epsilon
        s_ = relevance / z_
        c_ = tf.nn.conv2d_backprop_input(tf.shape(activation), w_neg, s_, strides, padding='SAME')
        return (alpha * c - beta * c_) * activation

    

#   def backprop_fc(self, name, activation, relevance):
#     w = self.act_weights[name]
#     w_pos = tf.maximum(0.0, w)
#     z = tf.matmul(activation, w_pos) + self.epsilon
#     s = relevance / z
#     c = tf.matmul(s, tf.transpose(w_pos))
#     return c * activation

#   def backprop_flatten(self, activation, relevance):
#     shape = activation.get_shape().as_list()
#     shape[0] = -1
#     return tf.reshape(relevance, shape)

#   def backprop_max_pool2d(self, activation, relevance, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
#     z = tf.nn.max_pool(activation, ksize, strides, padding='SAME') + self.epsilon
#     s = relevance / z
#     c = gen_nn_ops.max_pool_grad_v2(activation, z, s, ksize, strides, padding='SAME')
#     return c * activation

#   def backprop_conv2d(self, name, activation, relevance, strides=[1, 1, 1, 1]):
#     w = self.act_weights[name]
#     w_pos = tf.maximum(0.0, w)
#     z = tf.nn.conv2d(activation, w_pos, strides, padding='SAME') + self.epsilon
#     s = relevance / z
#     c = tf.nn.conv2d_backprop_input(tf.shape(activation), w_pos, s, strides, padding='SAME')
#     return c * activation

    
    

    def get_heatmap(self, samples, label, sess):
        #TODO: be able to access one specified label
        saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
        saver.restore(sess, tf.train.latest_checkpoint(logdir))


        heatmap = sess.run(self.relevances[-1], feed_dict={self.X: samples, self.Y: label})[0].reshape(im_size, im_size)
        heatmap /= heatmap.max()

        return heatmap, y_batch, _name

    def test_lrp(self):
        with tf.Session() as sess:
            self.test_filenames = tf.placeholder(tf.string, shape=[None])
            self.test_dataset = tf.data.TFRecordDataset(self.test_filenames).map(_parse_function).batch(1).repeat()
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            self.test_next_element = self.test_iterator.get_next()
            self.testing_filenames = ["data/testing_flat_x_156_full_dataset.tfrecords"]
            sess.run(self.test_iterator.initializer, feed_dict={self.test_filenames: self.testing_filenames})  


            saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
            saver.restore(sess, tf.train.latest_checkpoint(logdir))

            samples, y_batch, _name = sess.run(self.test_next_element)
            R = sess.run(self.relevances, feed_dict={self.X: samples, self.Y: y_batch})
            for r in R:
                print(r.sum())

    def test(self, sess):
        n_batches = 100

        avg_accuracy = 0
        for batch in range(n_batches):
            x_batch, y_batch, _name = sess.run(self.test_next_element)
            avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: x_batch, self.y: y_batch})[0]
        avg_accuracy /= n_batches
        print('Testing Accuracy {0:6.4f}'.format(avg_accuracy))


if __name__ == '__main__':
    lrp = LayerwiseRelevancePropagation()
    lrp.test_lrp()
    with tf.Session() as sess:
        
        #test_filenames = tf.placeholder(tf.string, shape=[None])
        test_dataset = tf.data.TFRecordDataset("data/testing_flat_x_156_full_dataset.tfrecords").map(_parse_function).batch(1)#repeat()
        test_iterator = test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()
        #testing_filenames = ["testing_flat_x_156_full_dataset.tfrecords"]
        #ess.run(test_iterator.initializer, feed_dict={test_filenames: testing_filenames})
        
        for i in range(500):
                samples, y_batch, _name = sess.run(test_next_element)
                
                heatmap_, label, _name = lrp.get_heatmap(samples, y_batch, sess)
                heatmap_ = heatmap_ayo(heatmap_)
                print(_name[0].decode(),i)
                if np.argmax(label) == 0:
                    label = 'CN'
                if np.argmax(label) == 1:
                    label = 'AD'
                if np.argmax(label) == 2:
                    label = 'EMCI'
                if np.argmax(label) == 3:
                    label = 'CMCI'
                if np.argmax(label) == 4:
                    label = 'MCI'
            
                save_name = _name[0].decode() + str('_x')
                
                title = str(label)
                fig = plt.figure()
                #plt.title(title)
                fig.tight_layout()
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax.imshow(heatmap_, interpolation='bilinear')
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig('{0}{1}.png'.format(resultsdir, save_name),bbox_inches='tight', pad_inches=0)
                plt.close()
                