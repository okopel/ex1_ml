# Ori Kopel
# 205533151
# K-means algorithm

import math
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.misc import imread

# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])
k_arr = [2, 4, 8, 16]
iter = 11
plt.imshow(A)
plt.grid(False)


# the first centroids
def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0.0, 0.0, 0.0],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0.0, 0.0],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


# get pixel from the image (RGB format) and the cectroids list
# return the closest centroid (By norm)
def find_closest_centroid(pixel, centroidsList):
    minDis = float("inf")
    minIndex = 0
    i = 0
    for cent in centroidsList:
        dis = distance(pixel, cent)
        if dis < minDis:
            minIndex = i
            minDis = dis
        i += 1
    return minIndex


# calculate the distace from point x to specific centroid
def distance(x, cent):
    r = (x[0] - cent[0]) * (x[0] - cent[0])
    g = (x[1] - cent[1]) * (x[1] - cent[1])
    b = (x[2] - cent[1]) * (x[2] - cent[1])
    return math.sqrt(r + g + b)


# get the pixels which closest to our centroid and rePlace the centroid
def calculateOneNewCentroid(pixelList):
    r = 0
    g = 0
    b = 0
    i = 0
    for pixel in pixelList:
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]
        i += 1
    # NOTE: there isnt i=0 because i entered to an empty list the original centroid
    r /= i
    g /= i
    b /= i
    return [r, g, b]


# cpecification of each centroid to the avg of his pixels set
def calculateNewCentoids(dictonary):
    newCenr = []
    for entry in dictonary:
        newCenr.append(calculateOneNewCentroid(dictonary[entry]))
    return newCenr


# one iteration in order to specific the centroids
def oneIter(centroidsList, k):
    dictonary = {}
    # init the dictionary to key:list of pixels
    for cent in range(k):
        dictonary[cent] = []
    # find the closest cent to the pixel
    for pixel in X:
        cloCent = find_closest_centroid(pixel, centroidsList)
        dictonary[cloCent].append(pixel)
    # enter original centroid to empty list
    for cent in range(k):
        if dictonary[cent] == []:
            dictonary[cent].append(centroidsList[cent])
    return calculateNewCentoids(dictonary)


# iteration of wanted K (# of centroids)
def oneOfK(k):
    centroidsList = init_centroids(X, k)
    for i in range(iter):
        print("iter {}:".format(i), end=' ')
        for j in range(k):
            r = floorNum(centroidsList[j][0])
            g = floorNum(centroidsList[j][1])
            b = floorNum(centroidsList[j][2])
            print("[{}, {}, {}]".format(r, g, b), end='')
            if j != (k - 1):
                print(", ", end='')
        print()
        centroidsList = oneIter(centroidsList, k)


# take 2 digit after the floating point
def floorNum(num):
    return numpy.floor(num * 100) / 100


# main
def main():
    for i in k_arr:
        print("k={}:".format(i))
        oneOfK(i)


# call the main
main()
