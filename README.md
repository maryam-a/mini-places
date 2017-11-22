# Mini Places Challenge
MIT 6.819 Mini Places Challenge by [Maryam Archie](https://github.com/maryam-a) and [Sandeep Silwal](https://github.com/ssilwa).

## Getting Started
1. To run these scripts, you first need to download the [image data](https://github.com/maryam-a/mini-places-data). Be warned - it is going to take a long time to download.

2. Edit the data paths in `augment_images.py` and `vgg16.py`.

3. Download and Python 3 (if you don't already - preferably from Anaconda)

4. Install the following dependencies:
```
pip install tensorflow
pip install tensorflow-gpu
pip install scipy
pip install pillow
```

5. It is strongly recommended that you use a GPU when running `vgg16.py`.

## Augmenting the image data set
With `augment_images.py`, you can:
- Increase the dataset from 100,000 to 400,000 images with label-preserving transformations
- Update `data\train.txt` with the new images
- Calculate the new mean of the data set

For your convenience, this has already been done and is available in the [mini-places-data](https://github.com/maryam-a/mini-places-data) repository. In addition, the data mean in `vgg16.py` has been updated to reflect this.

## References
- Original Databases: [Places2 Database](http://places2.csail.mit.edu), [Places1 Database](http://places.csail.mit.edu)
- [Mini Places Challege](https://github.com/CSAILVision/miniplaces) 