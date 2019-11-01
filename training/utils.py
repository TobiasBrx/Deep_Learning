import gzip
import pickle
import tensorflow as tf
import numpy as np

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature([156, 156, 1], tf.float32),
              "label": tf.FixedLenFeature((), tf.int64),
               'name': tf.FixedLenFeature((), tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features["image"]= tf.reshape(parsed_features["image"],[-1]) 
    parsed_features["label"]=tf.one_hot(parsed_features["label"],5)
    return parsed_features["image"], parsed_features["label"], parsed_features["name"]




def visualize(x,colormap):

        N = len(x); assert(N<=16)
        print("ok1")
        x = colormap(x/numpy.abs(x).max())
        print("ok2")
        # Create a mosaic and upsample
        x = x.reshape([1,N, 156, 156,3])
        x = numpy.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
        x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
        x = numpy.kron(x,numpy.ones([2,2,1]))
        
        return x

        


def heatmap_ayo(x):

        x = x[...,np.newaxis]

        # positive relevance
        hrp = 0.9 - np.clip(x-0.3,0,0.7)/0.7*0.5
        hgp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4
        hbp = 0.9 - np.clip(x-0.0,0,0.3)/0.3*0.5 - np.clip(x-0.3,0,0.7)/0.7*0.4

        # negative relevance
        hrn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
        hgn = 0.9 - np.clip(-x-0.0,0,0.3)/0.3*0.5 - np.clip(-x-0.3,0,0.7)/0.7*0.4
        hbn = 0.9 - np.clip(-x-0.3,0,0.7)/0.7*0.5

        r = hrp*(x>=0)+hrn*(x<0)
        g = hgp*(x>=0)+hgn*(x<0)
        b = hbp*(x>=0)+hbn*(x<0)

        return np.concatenate([r,g,b],axis=-1)
    
    
def heatmap(x):

	x = x[...,numpy.newaxis]

	# positive relevance
	hrp = 0.9 - numpy.clip(x-0.3,0,0.7)/0.7*0.5
	hgp = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4
	hbp = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4

	# negative relevance
	hrn = 0.9 - numpy.clip(-x-0.0,0,0.3)/0.3*0.5 - numpy.clip(-x-0.3,0,0.7)/0.7*0.4
	hgn = 0.9 - numpy.clip(-x-0.0,0,0.3)/0.3*0.5 - numpy.clip(-x-0.3,0,0.7)/0.7*0.4
	hbn = 0.9 - numpy.clip(-x-0.3,0,0.7)/0.7*0.5

	r = hrp*(x>=0)+hrn*(x<0)
	g = hgp*(x>=0)+hgn*(x<0)
	b = hbp*(x>=0)+hbn*(x<0)

	return numpy.concatenate([r,g,b],axis=-1)
