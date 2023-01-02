import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import sys
import math

if (len(sys.argv) != 2):
	print("Usage: build_histogram.py <# of bins>")
	sys.exit()

BINS = int(sys.argv[1])
THRESHOLD = 0.1

def image2histogram(pixels):
    #normalize to float
    if pixels.dtype == 'uint8':
        pixels = pixels.astype('float32')
        pixels /= 255.0
    
    hist = np.empty(shape=(BINS, 2))
    hist.fill(0)

    for x in range(BINS):
        hist[x,0] = x / BINS
    for pixel in np.nditer(pixels):
        bucket = math.floor(pixel*BINS)
        if bucket == BINS:
            hist[-1, 1] += 1
        else:
            hist[bucket, 1] += 1
    return hist

def thresholdimage(pixels):
    #normalize to float
    if pixels.dtype == 'uint8':
        pixels = pixels.astype('float32')
        pixels /= 255.0
    
    for (x,y), value in np.ndenumerate(pixels):
        if value < THRESHOLD:
            pixels[x,y] = 0.0
    skio.imshow(pixels)
    skio.show()
    skio.imsave("thresholded.png", pixels)


pixels = skio.imread("crowd.tif", plugin="tifffile")
skio.imshow(pixels)
skio.show()
hist_data = image2histogram(pixels)
plt.bar(x=hist_data[:,0], height=hist_data[:,1], align='edge', width=1/BINS)
plt.show()

thresholdimage(pixels)