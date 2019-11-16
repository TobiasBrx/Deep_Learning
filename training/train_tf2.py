import tensorflow as tf
import numpy as np
from model_tf2 import MRI, MRI_2dCNN
from utils_tf2 import ModelSelectionError
import time
#import pickle
logdir = 'logs_x_FFNN/'
chkpt = 'logs_x_FFNN/model.ckpt'
n_epochs = 10
batch_size = 5
label_size = 3
im_size=156
total_training_files = 12975
total_validation_files = 960
class Trainer:
    
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope() as strategy:
        
        def __init__(self, model='CNN', im_size=im_size):
            self.im_size = im_size
            try:
                if model =='FF':
                    self.model = MRI(train=True, im_size=im_size, label_size=label_size)
                elif model=='CNN':
                     self.model = MRI_2dCNN(train=True, im_size=im_size, label_size=label_size)
                elif (model != 'FF') | (model != 'CNN'):
                    raise ModelSelectionError()
            except ModelSelectionError:
                 print("Please select either FF for a Fully connected Feed-Forward network or CNN for a convolutional network.")
                    
                    
        @tf.function            
        def cost(self, y_pred, y_true):
            return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

        @tf.function 
        def accuracy(self, y_pred, y_true):
            equals = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
            return tf.reduce_mean(input_tensor=tf.cast(equals, tf.float32))


        def run(self, train_filenames, val_filenames, x_slice=None, y_slice=None, z_slice=None, strategy=strategy):
                self.checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer) 
                self.file_writer = tf.summary.create_file_writer(logdir)
                self.dataset = tf.data.TFRecordDataset(train_filenames).map(
                    _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
                    buffer_size=5).batch(batch_size)#.repeat()
                #self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                #self.dist_dataset = strategy.experimental_distribute_dataset(self.dataset)
                self.val_dataset = tf.data.TFRecordDataset(val_filenames).map(
                    _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
                    buffer_size=5).batch(batch_size)#.repeat()
                #self.val_dataset = self.val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                #self.dist_val_dataset = strategy.experimental_distribute_dataset(self.val_dataset)

                for epoch in range(n_epochs):
                    t_start = time.time()
                    self.train(epoch, x_slice, y_slice, z_slice, t_start)
                    final_accuracy = self.validate(x_slice, y_slice, z_slice).numpy()
                    print('\nEpoch {0:>3}|  Validation Accuracy: {1:4.2f}'.format(str(epoch + 1), final_accuracy))
                    self.checkpoint.save(chkpt)
                
                return final_accuracy
        
        
        def train(self, epoch, x_slice, y_slice, z_slice, t_start):
            #TODO: not satisfied with batch sampling. initial batches are very similar. RAMIFICATIONS ?
            n_batches = int(total_training_files/batch_size)
            batch = 0
            avg_cost = 0.0
            avg_accuracy = 0.0
            times = []
            epoch_time = t_start
            self.file_writer.set_as_default()
            for x_batch, y_batch, _name in self.dataset:
                t0 = time.time()
                batch += 1
                batch_accuracy, current_loss = self.train_step(
                    x_batch, y_batch, _name, x_slice, y_slice, z_slice, tf.constant(batch, dtype=tf.float32),
                    tf.constant(epoch, dtype=tf.float32), tf.constant(n_batches, dtype=tf.float32))
                avg_cost += current_loss
                avg_accuracy += batch_accuracy
                completion = batch / n_batches
                t1 = time.time()
                if batch == 1:
                    batch_time =  t1 - t_start
                else:
                    batch_time = t1 - t_start - epoch_time
                epoch_time = t1 - t_start
                print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
                print(('\rEpoch {0:>3} {1} {2:3.0f}% Cost: {3:6.4f}, Avg. Acc: {4:6.4f}, Current Acc: {5:.2f}, ' +
                       'elapsed: {6:.0f}s, current: {7:.2f}').format(
                    '#' + str(epoch + 1), print_str, completion * 100, avg_cost / (batch), avg_accuracy / (batch),
                    batch_accuracy, epoch_time, batch_time), end='')
                
                if batch == n_batches:
                    break
                times.append(batch_time)
            print(end=' ')

            #with open('times.pkl', 'wb') as f:
            #     pickle.dump(times, f)

            return avg_accuracy
        
        
        @tf.function
        def train_step(self, x_batch, y_batch, _name, x_slice, y_slice, z_slice, batch, epoch, n_batches):
            x_batch = tf.cast(x_batch, dtype=tf.float32)
            y_batch = tf.cast(y_batch, dtype=tf.float32)
            x_batch = tf.reshape(x_batch, [tf.shape(x_batch)[0], self.im_size, self.im_size, self.im_size])
            if x_slice:
                x_batch = x_batch[:, x_slice, : , :]
            elif y_slice:
                x_batch = x_batch[:, : , y_slice, : ]
            elif z_slice:
                x_batch = x_batch[:, : , : , z_slice]
            y_batch = tf.reshape(y_batch, [tf.shape(y_batch)[0], -1])
            _name = tf.reshape(_name, [tf.shape(x_batch)[0], -1])
            y_pred = self.model.forward(x_batch)
            current_loss = self.model.cost(y_pred, y_batch)
            tf.summary.scalar('Cost', current_loss, step=tf.dtypes.cast(epoch * n_batches + batch, dtype=tf.int64))
            batch_accuracy = self.accuracy(y_pred, y_batch)
            tf.summary.scalar('Accuracy', batch_accuracy, step=tf.dtypes.cast(epoch * n_batches + batch, dtype=tf.int64))
            self.model.backward(x_batch, y_batch)
            #self.file_writer.add_summary(summ, epoch * n_batches + batch)
            return batch_accuracy, current_loss

        @tf.function    
        def validate(self, x_slice, y_slice, z_slice):
            n_batches = int(total_validation_files/batch_size)
            batch = 0
            avg_accuracy = 0.0
            for x_batch, y_batch, _name in self.val_dataset:
                x_batch = tf.cast(x_batch, dtype=tf.float32)
                y_batch = tf.cast(y_batch, dtype=tf.float32)
                batch += 1
                x_batch = tf.reshape(x_batch, [tf.shape(x_batch)[0], self.im_size, self.im_size, self.im_size])
                if x_slice:
                    x_batch = x_batch[:, x_slice, : , :]
                elif y_slice:
                    x_batch = x_batch[:, : , y_slice, : ]
                elif z_slice:
                    x_batch = x_batch[:, : , : , z_slice]
                y_pred = self.model.forward(x_batch)
                y_batch = tf.reshape(y_batch, [tf.shape(y_batch)[0], -1])
                accuracy = self.accuracy(y_pred, y_batch)
                avg_accuracy += accuracy
                if batch == n_batches:
                    break
            avg_accuracy /= n_batches
            return avg_accuracy
        
        
@tf.function         
def _parse_function(example_proto):
            features = {"image": tf.io.FixedLenFeature([im_size, im_size, im_size], tf.float32),
                      "label": tf.io.FixedLenFeature((), tf.int64),
                       'name': tf.io.FixedLenFeature((), tf.string, default_value='')}
            parsed_features = tf.io.parse_single_example(serialized=example_proto, features=features)
            parsed_features["image"] = tf.reshape(parsed_features["image"], [-1]) 
            parsed_features["label"] = tf.one_hot(parsed_features["label"], label_size)
            return parsed_features["image"], parsed_features["label"], parsed_features["name"]
        
        
if __name__ == '__main__':
        train_filenames = "../data/training_flat_156_3d.tfrecords"
        val_filenames = "../data/validation_flat_156_3d.tfrecords"
        #Trainer(model='FF').run(train_filenames, val_filenames, x_slice=127)
        Trainer().run(train_filenames, val_filenames, x_slice=77)