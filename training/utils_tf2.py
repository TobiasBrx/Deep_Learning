import gzip
import pickle
import tensorflow as tf
import numpy as np
        
class ModelSelectionError(Exception):
    
    pass
    
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

def create_booleanmasks(dim_tuple=(52, 52), h=5, w=5, s=5, pad=False):
    """
	inputs
		dim_tuple specifies mask dimensions
		h & w specify the dimensions of the True subframe
		s specifies the stride

	output:
		list of boolean masks
    """  
    
    if pad:
        raise ValueError("function not built to perform padding")
    arrh = dim_tuple[0]
    arrw = dim_tuple[1]
    oh_cur=0
    ow_cur=0
    h_cur = oh_cur
    w_cur = ow_cur
    output=[]
    bean = 0
    _fl =True
    while _fl:
        if (w_cur+w >arrw ) or (h_cur+h >arrh):
            _fl=False
        else:
            tmp = np.zeros(dim_tuple)
            tmp[w_cur:w_cur+w,h_cur:h_cur+h]=1
            output.append(tmp)
            w_cur+=s
            if w_cur+w >arrw:
                w_cur=ow_cur
                h_cur+=s
                if w_cur+w >arrw:
                    _fl=False
                elif h_cur+h >arrh:
                    _fl=False
            bean+=1
    masks = [i.astype(bool) for i in output]
    return masks