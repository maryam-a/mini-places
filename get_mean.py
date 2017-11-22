import os
import numpy as np
from scipy import misc, ndimage

data_root = '../data/images/train/'

total_mean = np.array([0,0,0], 'float64')
for dirname, dirnames, filenames in os.walk(data_root):
    # print path to all subdirectories first.
    for filename in filenames:
        original_image_path = os.path.join(dirname, filename)
        image = misc.imread(original_image_path)
        npm =  np.mean(image)
        print (npm)
        total_mean += np.mean(npm)

print ('Finally: ' + total_mean)