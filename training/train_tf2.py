import tensorflow as tf
from model_tf2 import MRI, MRI_2dCNN
from utils_tf2 import ModelSelectionError
import time
logdir = 'logs_x_FFNN/'
chkpt = 'logs_x_FFNN/model.ckpt'
n_epochs = 3 
batch_size = 10
label_size = 3
class Trainer:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope() as scope:
        
        def __init__(self, model='FF', im_size=156):
            try:
                if model =='FF':
                    self.model = MRI(train=True, im_size=im_size, label_size=label_size)
                elif model=='CNN':
                     self.model = MRI_2dCNN(train=True, im_size=im_size, label_size=label_size)
                elif (model != 'FF') | (model != 'CNN'):
                    raise ModelSelectionError()
            except ModelSelectionError:
                 print("Please select either FF for a Fully connected Feed-Forward network or CNN for a convolutional network.")
                    
        def cost(self, y_pred, y_true):
            return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

        def accuracy(self, y_pred, y_true):
            equals = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
            return tf.reduce_mean(input_tensor=tf.cast(equals, tf.float32))

        def run(self, train_filenames, val_filenames, x_slice=None, y_slice=None, z_slice=None):
                self.checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer) 
                self.file_writer = tf.summary.create_file_writer(logdir)
                self.dataset = tf.data.TFRecordDataset(train_filenames).map(_parse_function).shuffle(
                    buffer_size=20).batch(batch_size)#.repeat()
                self.val_dataset = tf.data.TFRecordDataset(val_filenames).map(_parse_function).shuffle(
                    buffer_size=20).batch(batch_size)#.repeat()

                for epoch in range(n_epochs):
                    t_start = time.time()
                    self.train(epoch, x_slice, y_slice, z_slice, t_start)
                    final_accuracy = self.validate(epoch, x_slice, y_slice, z_slice)
                    self.checkpoint.save(chkpt)
                
                return final_accuracy

        def train(self, epoch, x_slice, y_slice, z_slice, t_start):
            #TODO: not satisfied with batch sampling. initial batches are very similar. RAMIFICATIONS ?
            n_batches = 1290
            batch = 0
            avg_cost = 0
            avg_accuracy = 0
            times = []
            self.file_writer.set_as_default()
            for x_batch, y_batch, _name in self.dataset:
                t0 = time.time()
                batch += 1
                x_batch = tf.reshape(x_batch, [x_batch.shape[0], 256, 256, 256])
                if x_slice:
                    x_batch = x_batch[:, x_slice, 50:206, 50:206]
                elif y_slice:
                    x_batch = x_batch[:, 50:206, y_slice, 50:206]
                elif z_slice:
                    x_batch = x_batch[:, 50:206, 50:206, z_slice]
                y_batch = tf.reshape(y_batch, [y_batch.shape[0], -1])
                _name = tf.reshape(_name, [x_batch.shape[0], -1])
                y_pred = self.model.forward(x_batch)
                current_loss = self.model.cost(y_pred, y_batch)
                tf.summary.scalar('Cost', current_loss, step=epoch * n_batches + batch)
                batch_accuracy = self.accuracy(y_pred, y_batch)
                tf.summary.scalar('Accuracy', batch_accuracy, step=epoch * n_batches + batch)
                self.model.backward(x_batch, y_batch)
                avg_cost += current_loss
                avg_accuracy += batch_accuracy
                #self.file_writer.add_summary(summ, epoch * n_batches + batch)
                completion = batch / n_batches
                t1 = time.time()
                batch_time = t1-t0
                epoch_time = t1 - t_start
                print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
                print(('\rEpoch {0:>3} {1} {2:3.0f}% Cost: {3:6.4f}, Avg. Accuracy: {4:6.4f}, Current Accuracy: {5:.2f}, ' +
                      'elapsed time: {6:.0f}s').format(
                      '#' + str(epoch + 1), print_str, completion * 100, avg_cost / (batch), avg_accuracy / (batch), 
                      batch_accuracy, epoch_time), end='')
                if batch == n_batches:
                    break
                times.append(batch_time)
            print(end=' ')
            #import pickle
            #with open('times.pkl', 'wb') as f:
            #     pickle.dump(times, f)

        def validate(self, epoch, x_slice, y_slice, z_slice):
            n_batches = 100
            batch = 0
            avg_accuracy = 0
            for x_batch, y_batch, _name in self.val_dataset:
                batch += 1
                x_batch = tf.reshape(x_batch, [x_batch.shape[0], 256, 256, 256])
                if x_slice:
                    x_batch = x_batch[:, x_slice, 50:206, 50:206]
                elif y_slice:
                    x_batch = x_batch[:, 50:206, y_slice, 50:206]
                elif z_slice:
                    x_batch = x_batch[:, 50:206, 50:206, z_slice]
                y_pred = self.model.forward(x_batch)
                y_batch = tf.reshape(y_batch, [y_batch.shape[0], -1])
                accuracy = self.accuracy(y_pred, y_batch)
                avg_accuracy += accuracy
                if batch == n_batches:
                    break
            avg_accuracy /= n_batches
            print('\nEpoch {0:>3}|  Validation Accuracy: {1:4.2f}'.format(str(epoch + 1), avg_accuracy))
            return float(avg_accuracy)
                

    if __name__ == '__main__':
        Trainer().run()
        
        
def _parse_function(example_proto):
            features = {"image": tf.io.FixedLenFeature([256, 256, 256], tf.float32),
                      "label": tf.io.FixedLenFeature((), tf.int64),
                       'name': tf.io.FixedLenFeature((), tf.string, default_value='')}
            parsed_features = tf.io.parse_single_example(serialized=example_proto, features=features)
            parsed_features["image"] = tf.reshape(parsed_features["image"], [-1]) 
            parsed_features["label"] = tf.one_hot(parsed_features["label"], label_size)
            return parsed_features["image"], parsed_features["label"], parsed_features["name"]