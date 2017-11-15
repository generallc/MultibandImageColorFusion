from __future__ import print_function, division
import os
import time
import cv2
import glob
import numpy as np
from scipy.misc import imread


def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])


def _index_generator(N, batch_size=32, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)


def subimage_generator(img, stride, patch_size, nb_images, mode=True):
    for _ in range(nb_images):
        for x in range(0, img.shape[0] - patch_size, stride):
            for y in range(0, img.shape[1] - patch_size, stride):
                if mode:
                    subimage = img[x: x + patch_size, y: y + patch_size]
                else:
                    subimage = img[x: x + patch_size, y: y + patch_size, :]
                yield subimage


def transform_images(directory, output_directory, mode=True, patch_size=128, stride=100):
    index = 1   # numbers for images
    name = 1    # numbers for patches

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        for f in glob.glob(output_directory + '*'):
            os.remove(f)

    # Calculate the numbers of the directory
    nb_images = len([img for img in os.listdir(directory)])
    if nb_images == 0:
        print("Extract the training images or images into a directory with the name 'XX_input_images'")
        exit()
    else:
        print("Transforming %d images." % nb_images)

    # Extract the image patches
    for file in sorted(os.listdir(directory)):
        img = imread(directory + file, mode='RGB')

        if mode:
            # RGB to Gray image
            if img.shape[-1] == 3:
                img = rgb2gray(img)

        # Create patches
        nb_row_images = (img.shape[0] - patch_size) // stride + 1
        nb_column_images = (img.shape[1] - patch_size) // stride + 1
        nb_images = nb_row_images * nb_column_images

        if mode:
            samples = np.empty((nb_images, patch_size, patch_size)).astype('uint8')
        else:
            samples = np.empty((nb_images, patch_size, patch_size, 3)).astype('uint8')

        # if the mode is true the extract gray patches, else extract color patches
        image_subsample_iterator = subimage_generator(img, stride, patch_size, nb_images, mode)
        stride_row_range = len(range(0, img.shape[0] - patch_size, stride))
        stride_column_range = len(range(0, img.shape[1] - patch_size, stride))

        i = 0
        for j in range(stride_row_range):
            for k in range(stride_column_range):
                if mode:
                    samples[i, :, :] = next(image_subsample_iterator)
                else:
                    samples[i, :, :, :] = next(image_subsample_iterator)
                i += 1
        t1 = time.time()

        # Save the extracted patches to the output directory
        for i in range(nb_images):

            # generate the patches save name
            patch_name = str(name)
            if len(patch_name) < 5:
                for o in range(5 - len(patch_name)):
                    patch_name = '0' + patch_name

            ip = samples[i]

            # when the patches is color, convert it to OpenCV style
            if not mode:
                ip = ip[:, :, ::-1]

            cv2.imwrite(output_directory + "%s.jpg" % patch_name, ip)
            name += 1
        print("Finished image %d in time %0.2f seconds. (%s)" % (index, time.time() - t1, file))
        index += 1

    print("Images transformed and generate %d patches. Saved at directory : %s" % (name-1, output_directory))
    return name-1


def generate_text(directory, nb_patches):

    all_list = range(nb_patches)
    all_list = [x + 1 for x in all_list]
    np.random.shuffle(all_list)

    # write the training image names
    with open(directory, "w") as f:

        new_list = all_list[0:320]
        for i in new_list:
            name = str(i)

            if len(name) < 5:
                for o in range(5 - len(name)):
                    name = '0' + name
                name = name + '.jpg 0\n'
            f.write(name)

    # write the development image names
    with open(directory, "a") as f:

        new_list = all_list[320:340]
        for i in new_list:
            name = str(i)
            if len(name) < 5:
                for o in range(5 - len(name)):
                    name = '0' + name
                name = name + '.jpg 1\n'
            f.write(name)

    # write the test image names
    with open(directory, "a") as f:

        new_list = all_list[340:360]
        for i in new_list:
            name = str(i)
            if len(name) < 5:
                for o in range(5 - len(name)):
                    name = '0' + name
                name = name + '.jpg 2\n'
            f.write(name)


if __name__ == "__main__":

    # the original image data path and output patch path
    II_input_path = "../../data/raw/original/II/"
    IR_input_path = "../../data/raw/original/IR/"
    Vis_input_path = "../../data/raw/original/Vis/"

    Color_input_path = "../../data/raw/original/Color/"

    IIPatch_output_path = "../../data/raw/patches/IIPatches/"
    IRPatch_output_path = "../../data/raw/patches/IRPatches/"
    VisPatch_output_path = "../../data/raw/patches/VisPatches/"
    ColorPatch_output_path = "../../data/raw/patches/ColorPatches/"

    PatchSize = 128
    Stride = 60

    # if the mode is true the extract gray patches, else extract color patches

    # '''
    # Extract the II patches
    # '''
    # transform_images(II_input_path, IIPatch_output_path, mode=True, patch_size=PatchSize, stride=Stride)
    #
    # '''
    # Extract the IR patches
    # '''
    # transform_images(IR_input_path, IRPatch_output_path, mode=True, patch_size=PatchSize, stride=Stride)
    #
    # '''
    # Extract the Vis patches
    # '''
    # transform_images(Vis_input_path, VisPatch_output_path, mode=True, patch_size=PatchSize, stride=Stride)
    #
    #
    #
    # '''
    # Extract the Color patches
    # '''
    # nb_patches = transform_images(Color_input_path, ColorPatch_output_path, mode=False, patch_size=PatchSize, stride=Stride)
    #
    # '''
    # Genarate the list_datasets   List of dataset names. Choose training, validation or test
    # '''

    generate_text("../../data/raw/list_datasets.txt", 360)

    pass










