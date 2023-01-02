import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import img_as_float, img_as_ubyte
from skimage import exposure
from skimage.morphology import rectangle
from skimage.filters import rank
import sys

if (len(sys.argv) != 2):
	print("Usage: histogram_equalization.py <# of bins>")
	sys.exit()

BINS = int(sys.argv[1])

def color2gray(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

def display_histogram(image):
    # Display histogram
    histogram, bin_edges = np.histogram(image, bins=BINS, range=(0, 1))
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.plot(bin_edges[0:-1], histogram)
    plt.show()

pixels = skio.imread("xray.png")
pixels = color2gray(img_as_float(pixels))
display_histogram(pixels)
# Equalization
#Params bins, mask(same shape as image)
img_eq = exposure.equalize_hist(pixels, nbins=BINS)
display_histogram(img_eq)
skio.imshow(img_eq)
skio.imsave("img_eq.png", img_eq)

#Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(image=pixels, clip_limit=0.01, nbins=BINS)
display_histogram(img_adapteq)
skio.imshow(img_adapteq)
skio.imsave("img_adapteq.png", img_adapteq)

# Local Equalization
footprint = rectangle(583, 393)
print(pixels.shape)
img_localeq = rank.equalize(img_as_ubyte(pixels), footprint)
display_histogram(img_as_float(img_localeq))
skio.imshow(img_localeq)
skio.imsave("img_localeq.png", img_localeq)