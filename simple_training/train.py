import tensorflow as tf
from utils import _parse_function
from model import ThreeDNet, im_size, modified_size, output_size

logdir = 'logs_3D_CNN/'
chkpt = 'logs_3D_CNN/model.ckpt'

n_epochs = 30 # zdirection requires more?
batch_size = 64

buffer_size_training_set = 12400
buffer_size_validation_set = 1000
buffer_size_test_set = 1000

class Trainer:
    

    def __init__(self):

        with tf.variable_scope('3D_CNN'):
            
            self.X = tf.placeholder(tf.float32, [None, modified_size, modified_size, modified_size], name='X')
            self.y = tf.placeholder(tf.float32, [None, output_size], name='y')
            print("input placeholder ok")
            model  = ThreeDNet(self.X, 0.5, output_size)
            score = model.y_conv
            softmax = tf.nn.softmax(score)
            
            vars_ = tf.trainable_variables() 
            self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_]) * 0.001
            
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.y)) + self.lossL2
            self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
            self.preds = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))

        self.cost_summary = tf.summary.scalar(name='Cost', tensor=self.cost)
        
        self.accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=self.accuracy)

        self.summary = tf.summary.merge_all()

    def run(self):
        #TODO: MAKE IT WORK WITH GPUs
        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:

            
            sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
            
            self.filenames = tf.placeholder(tf.string, shape=[None])
            #self.dataset = tf.data.TFRecordDataset(self.filenames).map(_parse_function).shuffle(buffer_size=buffer_size_validation_set).batch(batch_size).repeat()
            
            """NOT SHUFFLING TO GAIN TIME"""
            self.dataset = tf.data.TFRecordDataset(self.filenames).map(_parse_function).batch(batch_size).repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()
            self.training_filenames = ["../create_tfrecords/validation_flat_256_3d.tfrecords"]
            sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})
                    
            self.val_filenames = tf.placeholder(tf.string, shape=[None])
            #self.val_dataset = tf.data.TFRecordDataset(self.val_filenames).map(_parse_function).shuffle(buffer_size=buffer_size_test_set).batch(batch_size).repeat()
            """NOT SHUFFLING TO GAIN TIME"""
            self.val_dataset = tf.data.TFRecordDataset(self.val_filenames).map(_parse_function).batch(batch_size).repeat()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
            self.val_next_element = self.val_iterator.get_next()
            self.validation_filenames = ["../create_tfrecords/testing_flat_256_3d.tfrecords"]
            sess.run(self.val_iterator.initializer, feed_dict={self.val_filenames: self.validation_filenames})


            for epoch in range(n_epochs):
                self.train(sess, epoch)
                self.validate(sess)
                self.saver.save(sess, chkpt)

    def train(self, sess, epoch):
        import time
        #TODO: access full size of dataset
        n_batches=1240
        avg_cost = 0
        avg_accuracy = 0
        times = 0
        time_10batches=0
        for batch in range(n_batches):
            t0 = time.time()
            x_batch, y_batch, _name = sess.run(self.next_element)
            _, batch_cost, batch_accuracy, summ = sess.run([self.optimizer, self.cost, self.accuracy, self.summary], feed_dict={self.X: x_batch, self.y: y_batch})
            avg_cost += batch_cost
            avg_accuracy += batch_accuracy
            self.file_writer.add_summary(summ, epoch * n_batches + batch)

            completion = batch / n_batches
            print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
            print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1), 
                print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
            t1 = time.time()
            batch_time = t1-t0
            times+=batch_time
            time_10batches+=batch_time
            if (batch+1)%10:
                print(" 10 batches took " + str(time_10batches))
                time_10batches = 0
        print(end=' ')
        print("Epoch took " + str(times))
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
