import cv2
from skimage import color



img_path ='/media/generallc/DoctoralResearch/PYTHONCODE/CNN_autoencoder/Colorization/' \
          'Colorful_fusion/data/raw/color_bqgc/00045.jpg'
size=64


img_color = cv2.imread(img_path)
img_color = img_color[:, :, ::-1]
img_black = cv2.imread(img_path, 0)

img_color = cv2.resize(img_color, (size, size), interpolation=cv2.INTER_AREA)
img_black = cv2.resize(img_black, (size, size), interpolation=cv2.INTER_AREA)

img_lab = color.rgb2lab(img_color)

img_lab = img_lab.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)
img_color = img_color.reshape((1, size, size, 3)).transpose(0, 3, 1, 2)
img_black = img_black.reshape((1, size, size, 1)).transpose(0, 3, 1, 2)

