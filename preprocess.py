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
image_size = 28
pixel_depth = 255.0
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


def load_letter(letter_dir, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(letter_dir)
    # (num image, image width, image height)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    print(letter_dir)
    for image in image_files:
        image_file = os.path.join(letter_dir, image)
        try:
            # normalize image to [-0.5, 0.5]
            image_data = (ndimage.imread(image_file).astype(float) \
                - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, "- it's ok, skipping.")

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d'
                        % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_dirs, min_num_images_per_class, force=False):
    dataset_names = []
    for d in data_dirs:  # A to J
        set_filename = d + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(d, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names


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


def draw_datasets(pickle_file):
    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        print(letter_set.shape)
    num_rows = num_cols = 10
    pos = 1
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows, num_cols, pos)
            image = letter_set[random.randint(0, len(letter_set))]
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

    # draw_images(train_dirs)

    train_datasets = maybe_pickle(train_dirs, 45000)
    test_datasets = maybe_pickle(test_dirs, 1800)

    print('train_datasets:', train_datasets)
    print('test_datasets:', test_datasets)

    draw_datasets(train_datasets[0])
