from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import tarfile
import pickle
from scipy import ndimage, misc
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve


url = 'http://yaroslavvb.com/upload/notMNIST/'
num_classes = 10  # A to J
np.random.seed(133)


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify' + filename + '.Can you get to it with a browser?'
        )
    return filename


def maybe_extract(filename, force=False):
    """Extract tar.gz"""
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_dirs = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if len(data_dirs) != num_classes:
        raise Exception(
            'Expected %d dirs, one per class. Found % instead.' % (
                num_classes, len(data_dirs))
        )
    return data_dirs


def draw_images(root_dir):
    """Draw sample images for each class"""
    assert len(root_dir) == num_classes  # A to J
    num_cols = 10
    pos = 1
    for i in range(num_classes):
        target_dir = root_dir[i]
        for j in range(num_cols):
            plt.subplot(num_classes, num_cols, pos)
            random_file = random.choice(os.listdir(target_dir))
            image = misc.imread(os.path.join(target_dir, random_file))
            plt.imshow(image, cmap=plt.cm.gray)
            plt.axis('off')
            pos += 1
    plt.show()


if __name__ == '__main__':
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    print('train_filename:', train_filename)
    print('test_filename:', test_filename)

    train_dirs = maybe_extract(train_filename)
    test_dirs = maybe_extract(test_filename)

    print('train_dirs:', train_dirs)
    print('test_dirs:', test_dirs)

    draw_images(train_dirs)

