import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import sys
from skimage import img_as_float
from skimage.segmentation import flood_fill
from skimage.filters import threshold_otsu
from skimage.morphology import flood_fill

if (len(sys.argv) != 3):
	print("Usage: components.py <# of bins for histogram> <min component size")
	sys.exit()

BINS = int(sys.argv[1])
CONNECTED_COMPONENT_SIZE = int(sys.argv[2])
ADJACENT_PIXEL_DISTANCE = 1


def random_color():
    random_r = np.random.random()
    random_g = np.random.random()
    random_b = np.random.random()
    return [random_r, random_g, random_b]


def display_histogram(image):
    # Display histogram
    threshold = threshold_otsu(image)
    histogram, bin_edges = np.histogram(image, bins=BINS, range=(0, 1))
    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.axvline(threshold, color='r')
    plt.plot(bin_edges[0:-1], histogram)
    plt.show()

def thresholdimage(image):
    threshold = threshold_otsu(image)
    binary = img_as_float(image > threshold)
    plt.imshow(binary, cmap="gray")
    plt.show()
    return binary

def floodfill(image):
    rgb_image = np.stack((image,)*3, axis=-1)
    prev_image = np.copy(image)
    for (x,y), value in np.ndenumerate(image):
        if value == 0:
            #mark connected component with value 0.5
            flood_fill(image, (x,y), 0.5, in_place=True, connectivity=ADJACENT_PIXEL_DISTANCE)
            mask = np.subtract(image, prev_image)
            prev_image = np.copy(image)
            component_size = np.count_nonzero(mask == 0.5)
            if component_size > CONNECTED_COMPONENT_SIZE:
                #Color component with random color
                colored_component = np.stack((mask,)*3, axis=-1)
                random_rgb_color = random_color()
                for intensity,i in zip(random_rgb_color,range(3)):
                    colored_component[...,i] *= intensity * 2
                rgb_image += colored_component
            else:
                #Remove small components(Turn them white)
                small_component = np.stack((mask,)*3, axis=-1)
                white_color = [1 , 1, 1]
                for intensity,i in zip(white_color,range(3)):
                    small_component[...,i] *= intensity * 2
                rgb_image += small_component
    return rgb_image

pixels = skio.imread("turkeys.tif", plugin="tifffile")
pixels = img_as_float(pixels)
display_histogram(pixels)
thresholded_image = thresholdimage(pixels)
connected_components_image = floodfill(thresholded_image)
plt.imshow(connected_components_image)
plt.show()
plt.imsave("connected_components.png", connected_components_image)