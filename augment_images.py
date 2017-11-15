import os
import numpy as np
from scipy import misc, ndimage

data_root = '../data/images/train/'
label_root = 'data/'
label_file = open(label_root + 'new_train.txt', 'w')
ROTATE = 2000
CROP = 3000
BLUR = 4000
    
def get_save_path(dirname, filename, distort_type):
    img_number = int(filename.split('.')[0])
    new_filename = "%08d" % (distort_type + img_number,) + ".jpg"
    return dirname + '/' + new_filename

def rotate_image(img):
    return ndimage.rotate(img, 90)

def crop_image(img):
    lx, ly, lz = img.shape
    return img[lx // 4: - lx // 4, ly // 4: - ly // 4]

def blur_image(img):
    return ndimage.gaussian_filter(img, 2)

# note - need to change ..data/images/train to just train -> problems with windows because of \
#      - replace \ with /
def record_label(image_path, i):
    label_line = image_path + " " + str(i)
    label_file.write(label_line + '\n')

def save_and_record_image(img, save_path, i):
     misc.imsave(save_path, img) # uses the Image module (PIL)
     record_label(save_path, i)

i = 0
for dirname, dirnames, filenames in os.walk(data_root):
    # print path to all subdirectories first.
    for filename in filenames:
        if filename == '00002001.jpg':
            print('###########################################')
            break
        print (dirname, filename)
        original_image_path = os.path.join(dirname, filename)
        image = misc.imread(original_image_path)
        save_and_record_image(rotate_image(image), get_save_path(dirname, filename, ROTATE), i)
        save_and_record_image(crop_image(image), get_save_path(dirname, filename, CROP), i)
        save_and_record_image(blur_image(image), get_save_path(dirname, filename, BLUR), i)
    i += 1

label_file.close()
        