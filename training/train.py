import tensorflow as tf
from utils import _parse_function
from model import MRI, MRI_2dCNN, im_size, im_size_squared, output_size

logdir = 'logs_x_FFNN/'
chkpt = 'logs_x_FFNN/model.ckpt'
n_epochs = 20 # zdirection requires more?
batch_size = 10


class Trainer:

    def __init__(self):

        with tf.variable_scope('FFNN_x'):
            
            self.model = MRI(train=True)

            self.X = tf.placeholder(tf.float32, [None, im_size_squared], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_size], name='y')

            self.activations, self.logits = self.model(self.X, self.y)

            tf.add_to_collection('LayerwiseRelevancePropagation', self.X)
            tf.add_to_collection('LayerwiseRelevancePropagation', self.y)
            
            for act in self.activations:
                tf.add_to_collection('LayerwiseRelevancePropagation', act)
            #TODO: Fix this L2 loss function
            #self.l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in self.model.params if 'b' not in p.name]) * 0.001
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)) # + self.l2_loss
            #self.optimizer = tf.train.AdamOptimizer().minimize(self.cost, var_list=self.model.params)
            self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
            self.preds = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))

        self.cost_summary = tf.summary.scalar(name='Cost', tensor=self.cost)
        
        self.accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=self.accuracy)

        self.summary = tf.summary.merge_all()

    def run(self):
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
            
            self.filenames = tf.placeholder(tf.string, shape=[None])
            self.dataset = tf.data.TFRecordDataset(self.filenames).map(_parse_function).shuffle(buffer_size=12400).batch(batch_size).repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()
            self.training_filenames = ["data/training_flat_x_156_full_dataset.tfrecords"]
            sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})
                    
            self.val_filenames = tf.placeholder(tf.string, shape=[None])
            self.val_dataset = tf.data.TFRecordDataset(self.val_filenames).map(_parse_function).shuffle(buffer_size=1000).batch(batch_size).repeat()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
            self.val_next_element = self.val_iterator.get_next()
            self.validation_filenames = ["data/validation_flat_x_156_full_dataset.tfrecords"]
            sess.run(self.val_iterator.initializer, feed_dict={self.val_filenames: self.validation_filenames})


            for epoch in range(n_epochs):
                self.train(sess, epoch)
                self.validate(sess)
                self.saver.save(sess, chkpt)

    def train(self, sess, epoch):
#         import time
        #TODO: access full size of dataset
        n_batches=1240
        avg_cost = 0
        avg_accuracy = 0
#         times = []
        for batch in range(n_batches):
#             t0 = time.time()
            x_batch, y_batch, _name = sess.run(self.next_element)
            _, batch_cost, batch_accuracy, summ = sess.run([self.optimizer, self.cost, self.accuracy, self.summary], feed_dict={self.X: x_batch, self.y: y_batch})
            avg_cost += batch_cost
            avg_accuracy += batch_accuracy
            self.file_writer.add_summary(summ, epoch * n_batches + batch)

            completion = batch / n_batches
            print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
            print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1), 
                print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
            #t1 = time.time()
            #batch_time = t1-t0
            #times.append(batch_time)
        print(end=' ')
#         import pickle
#         with open('times.pkl', 'wb') as f:
#             pickle.dump(times, f)
        



    def validate(self, sess):
        #TODO: access full size of dataset
        n_batches = 100
        avg_accuracy = 0
        for batch in range(n_batches):
            x_batch, y_batch, _name = sess.run(self.val_next_element)
            avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: x_batch, self.y: y_batch})[0]

        avg_accuracy /= n_batches
        print('Validation Accuracy {0:6.4f}'.format(avg_accuracy))

if __name__ == '__main__':
    Trainer().run()
