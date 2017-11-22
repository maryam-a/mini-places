#Authors - Sandeep Silwal and Maryam Archie
import os
import numpy as np
from scipy import misc, ndimage

# File paths - MODIFY PATHS ACCORDINGLY
data_root = '../data/images/train/'
label_root = 'data/'

# Constant Values
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

def save_disorted_image(img, save_path):
    misc.imsave(save_path, img)

def augment_images():
    print('Augmenting images')

    for dirname, dirnames, filenames in os.walk(data_root):
        for filename in filenames:
            print (dirname, filename)
            # retrieve the image path
            original_image_path = os.path.join(dirname, filename)
            # read the image
            image = misc.imread(original_image_path)
            # distort image and save in same location (with a different name)
            save_disorted_image(rotate_image(image), get_save_path(dirname, filename, ROTATE))
            save_disorted_image(crop_image(image), get_save_path(dirname, filename, CROP))
            save_disorted_image(blur_image(image), get_save_path(dirname, filename, BLUR))

    print('Completed augmenting imaged!')

def update_train_file():
    print ('Updating the training file')
    category_file = open(label_root + 'categories.txt', 'r')
    label_file = open(label_root + 'train.txt', 'w')

    categories = category_file.readlines()

    for c in categories:
        sc = c.split(" ")
        for i in range(1, 1001):
            label_file.write('train' + sc[0] + "/" + "%08d" % (i,)+ ".jpg " + sc[1])
        for j in range(2001, 5001):
            label_file.write('train' + sc[0] + "/" + "%08d" % (j,)+ ".jpg " + sc[1])

    label_file.close()
    category_file.close()
    print ('Completed updating the training file')

def get_new_data_mean():
    print ('Computing the new data mean')

    print ('The new data mean is ')

# Augment the images and update the training file
augment_images()
update_train_file()
get_new_data_mean()