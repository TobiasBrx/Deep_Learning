import tensorflow as tf

im_size = 256
modified_size = 156
output_size=5


class ThreeDNet(object):
    """alexNet model"""
    def __init__(self, x, keep_rate, classNum):
        print("Initilization of the model")
        self.X = tf.reshape(x, [-1, modified_size, modified_size, modified_size, 1], name="input")
        self.classNum = classNum
        self.keep_rate=keep_rate
        self.buildCNN()


    def buildCNN(self):
        with tf.name_scope("layer_a"):
            # conv => 16*16*16
            conv1 = tf.layers.conv3d(inputs=self.X, filters=16, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print('conv1')
            print(conv1)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print('conv2')
            print(conv2)
            # pool => 8*8*8
            pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            print('pool3')
            print(pool3)

        with tf.name_scope("layer_c"):
            # conv => 8*8*8
            conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print('conv4')
            print(conv4)
            # conv => 8*8*8
            conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            # pool => 4*4*4
            print('conv5')
            print(conv5)
            pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
            print('pool6')
            print(pool6)
            
        with tf.name_scope("batch_norm"):
            cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)
            print('cnn3d_bn')
            print(cnn3d_bn)
            
        with tf.name_scope("fully_con"):
            flattening = tf.reshape(cnn3d_bn, [-1, 2*2*2*128])
            dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
            # (1-keep_rate) is the probability that the node will be kept
            dropout = tf.layers.dropout(inputs=dense, rate=self.keep_rate, training=True)

        with tf.name_scope("y_conv"):
            self.y_conv = tf.layers.dense(inputs=dropout, units=self.classNum)
    
    


