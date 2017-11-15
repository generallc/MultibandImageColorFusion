# -*- coding: utf-8 -*-
from __future__ import print_function, division
from itertools import izip     #izip for python2; zip for python3

from keras.preprocessing.image import ImageDataGenerator
import os
import glob


def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename,
                  os.path.join(dir, titlePattern % title + ext))



directory = os.path.dirname(os.path.dirname(os.getcwd()))
data_gen_args = dict(rescale=1. / 255, rotation_range=90., horizontal_flip=True)

II_datagen = ImageDataGenerator(**data_gen_args)
IR_datagen = ImageDataGenerator(**data_gen_args)
color_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

II_generator = II_datagen.flow_from_directory(
    directory + '/data/raw/patches/IIPatches',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=378,
    class_mode=None,
    save_to_dir=directory + "/data/raw/patches/II_generator",
    save_prefix='',
    seed=seed)


IR_generator = IR_datagen.flow_from_directory(
    directory + '/data/raw/patches/IRPatches',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=378,
    class_mode=None,
    save_to_dir=directory + "/data/raw/patches/IR_generator",
    save_prefix='',
    seed=seed)

Vis_generator = IR_datagen.flow_from_directory(
    directory + '/data/raw/patches/VisPatches',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=378,
    class_mode=None,
    save_to_dir=directory + "/data/raw/patches/Vis_generator",
    save_prefix='',
    seed=seed)


color_generator = color_datagen.flow_from_directory(
    directory + '/data/raw/patches/ColorPatches',
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=378,
    class_mode=None,
    save_to_dir=directory + "/data/raw/patches/Color_generator",
    save_prefix='',
    seed=seed)

# combine generators into one which yields image and masks
# train_generator = izip(II_generator, IR_generator, Vis_generator, color_generator)


# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()
# train_generator.next()


print("amplification over")


# 对增强的数据进行重新命名,确保符合00001规范

II_path = directory + "/data/raw/patches/II_generator"
IR_path = directory + "/data/raw/patches/IR_generator"
Vis_path = directory + "/data/raw/patches/Vis_generator"
Color_path = directory + "/data/raw/patches/Color_generator"

IIfiles = sorted(os.listdir(II_path))
IRfiles = sorted(os.listdir(IR_path))
Visfiles = sorted(os.listdir(Vis_path))
Colorfiles = sorted(os.listdir(Color_path))


i = 1
for IIfile, IRfile, Visfile, Colorfile in zip(IIfiles, IRfiles,Visfiles, Colorfiles):

    print(IIfile, IRfile, Visfile, Colorfile)

    name = str(i)
    if len(name) < 5:
        for o in range(5 - len(name)):
            name = '0' + name
        name = name + '.jpg'
    os.rename(os.path.join(II_path, IIfile), os.path.join(II_path, name))
    os.rename(os.path.join(IR_path, IRfile), os.path.join(IR_path, name))
    os.rename(os.path.join(Vis_path, Visfile), os.path.join(Vis_path, name))
    os.rename(os.path.join(Color_path, Colorfile), os.path.join(Color_path, name))
    i = i+1
