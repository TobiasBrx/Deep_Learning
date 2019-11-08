import tensorflow as tf
from utils_tf2 import _parse_function
from model_tf2 import MRI, MRI_2dCNN, im_size, im_size_squared, output_size
import argparse

# logdir = 'logs_x_FFNN/'
# chkpt = 'logs_x_FFNN/model.ckpt'
# n_epochs = 10 # zdirection requires more?
# batch_size = 10

parser = argparse.ArgumentParser(description='This is to make command line runs easier')
parser.add_argument('--logdir', action='store', type=str, dest='logdir', nargs="?", default='logs_x_FFNN/')
parser.add_argument('--chkpt', action='store', type=str, dest='chkpt', nargs="?", default='logs_x_FFNN/model.ckpt')
parser.add_argument('--n_epochs', action='store', type=int, dest='chkpt', nargs="?", default=10)
parser.add_argument('--batch_size', action = 'store', type = int, dest = 'batch_size', nargs="?", default = 10)


# im_size = 156
# im_size_squared = im_size**2
# label_size = 5
# output_size=5
parser.add_argument('im_size', action='store', type=int, dest='im_size', nargs='?', default=156)
parser.add_argument('label_size', action='store', type=int, dest='label_size', nargs='?', default=5)
parser.add_argument('output_size', action='store', type=int, dest='output_size', nargs='?', default=5)
parser.add_argument('model_name', action='store', type=str, dest='model_name', nargs="?", default="MRI")
parser.add_argument('device', action='store', type=str, dest='device', nargs="?", default=None)
parser.add_argument('x_slice', action='store',type=int, dest='x_slice', default=None)
parser.add_argument('y_slice', action='store',type=int, dest='y_slice', default=None)
parser.add_argument('z_slice', action='store',type=int, dest='z_slice', default=None)
args=parser.parse_args()

class Trainer:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope() as scope:
        def __init__(self, logdir = args.logdir,
                            chkpt = args.chkpt,
                            n_epochs = args.n_epochs,
                            batch_size = args.batch_size,

                            im_size=args.im_size,
                            im_size_squared = args.im_size **2,
                            label_size = args.label_size,
                            output_size = args.output_size,
                            model_name = args.model_name,
                            device = args.device


                            ):

            #with tf.compat.v1.variable_scope('FFNN_x'):
            

            self.model = MRI(name = model_name, train=True, 
                            im_size=im_size, 
                            im_size_squared=im_size_squared, 
                            label_size =label_size, 
                            output_size=output_size,
                            device=device)

                #X: shape im_size_squared
                #y: shape output_size

                #tf.compat.v1.add_to_collection('LayerwiseRelevancePropagation', self.X)
                #tf.compat.v1.add_to_collection('LayerwiseRelevancePropagation', self.y)

                #for act in self.activations:
                #    tf.compat.v1.add_to_collection('LayerwiseRelevancePropagation', act)
                #TODO: Fix this L2 loss function
                #self.l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in self.model.params if 'b' not in p.name]) * 0.001


        def cost_summary(self, y_pred, y_true):
            return tf.compat.v1.summary.scalar(name='Cost', tensor=cost(X, y))

        def accuracy_summary(self, y_pred, y_true):
            return tf.compat.v1.summary.scalar(name='Accuracy', tensor=accuracy(y_pred, y_true))

        #def summary(self):
        #    return tf.compat.v1.summary.merge_all()

        def cost(self, y_pred, y_true):
            return tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

        #def preds(self, y_pred, y_true):
        #    return tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))

        def accuracy(self, y_pred, y_true):
            equals = tf.equal(tf.argmax(input=y_pred, axis=1), tf.argmax(input=y_true, axis=1))
            return tf.reduce_mean(input_tensor=tf.cast(equals, tf.float32))

        def run(self, train_filenames, val_filenames, x_slice=None, y_slice=None, z_slice=None):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(0.00001)
                self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer)#optimizer=self.optimizer, 
                self.file_writer = tf.summary.create_file_writer(logdir)
                self.dataset = tf.data.TFRecordDataset(train_filenames).map(_parse_function).shuffle(
                    buffer_size=20).batch(batch_size).repeat()
                self.val_dataset = tf.data.TFRecordDataset(val_filenames).map(_parse_function).shuffle(
                    buffer_size=20).batch(batch_size).repeat()

                for epoch in range(n_epochs):
                    self.train(epoch, x_slice, y_slice, z_slice)
                    self.validate()
                    self.checkpoint.save(chkpt)

        def train(self, epoch, x_slice, y_slice, z_slice):
    #         import time
            #TODO: access full size of dataset
            n_batches=1240
            avg_cost = 0
            avg_accuracy = 0
    #         times = []
            self.file_writer.set_as_default()

            for batch in range(n_batches):
        #         t0 = time.time()
                for x_batch, y_batch, _name in self.dataset: #
                    x_batch = tf.reshape(x_batch, [batch_size, 256, 256, 256])
                    if x_slice:
                        x_batch = x_batch[:, x_slice, 50:206, 50:206]
                        x_batch = tf.reshape(x_batch, [batch_size, -1])
                    elif y_slice:
                        x_batch = x_batch[:, 50:206, y_slice, 50:206]
                        x_batch = tf.reshape(x_batch, [batch_size, -1])
                    elif z_slice:
                        x_batch = x_batch[:, 50:206, 50:206, z_slice]
                        x_batch = tf.reshape(x_batch, [batch_size, -1])
                    y_batch = tf.reshape(y_batch, [batch_size, 5])
                    _name = tf.reshape(_name, [batch_size, -1])
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




        def validate(self):
            #TODO: access full size of dataset
            n_batches = 100
            avg_accuracy = 0
            for batch in range(n_batches):
                for x_batch, y_batch, _name in self.val_dataset:
                    avg_accuracy += accuracy(x_batch, y_batch)[0]

            avg_accuracy /= n_batches
            print('Validation Accuracy {0:6.4f}'.format(avg_accuracy))

# Jack moved this indentation outward
if __name__ == '__main__':
    slice_d = {0:'x_slice', 1:'y_slice', 2:'z_slice'}
    for e,i in enumerate([args.x_slice, args.y_slice, args.z_slice]):
        if i:
            print("only using {} of {}".format(i, slice_d))
    Trainer().run(x_slice=args.x_slice, y_slice=args.y_slice, z_slice=args.z_slice)


