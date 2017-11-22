#Authors - Sandeep Silwal and Maryam Archie
import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])
        # ensures that images aren't permutated and that labels aren't required
        self.test_data = kwargs['test_data'] 

        # read data info from lists
        self.list_im = []
        if not self.test_data:
            self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                if not self.test_data:
                    path, lab =line.rstrip().split(' ')
                else:
                    path =line.rstrip()
                self.list_im.append(os.path.join(self.data_root, path))
                if not self.test_data:
                    self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        if not self.test_data:
            self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        if not self.test_data:
            perm = np.random.permutation(self.num) 
            self.list_im[:, ...] = self.list_im[perm, ...]
            self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        if not self.test_data:
            labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            if not self.test_data:
                labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        if not self.test_data:
            return images_batch, labels_batch
        else:
            return images_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0