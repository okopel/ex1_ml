# Ori Kopel
# 205533151
# K-means algorithm


from copy import deepcopy

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])

k_arr = [2, 4, 8, 16]
numOfIters = 11


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
        return np.asarray([[0., 0., 0.],
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


# main
def main():
    for i in k_arr:
        print("k={}:".format(i))
        newCentList = oneOfK(i)
        showThePhoto(newCentList)


# iteration of wanted K (# of centroids)
def oneOfK(k):
    centroidsList = init_centroids(X, k)
    for i in range(numOfIters):
        print("iter {}:".format(i), end=' ')
        for j in range(k):
            r = np.floor((centroidsList[j][0]) * 100) / 100
            g = np.floor((centroidsList[j][1]) * 100) / 100
            b = np.floor((centroidsList[j][2]) * 100) / 100
            print("[{}, {}, {}]".format(r, g, b), end='')
            if j != (k - 1):
                print(", ", end='')
        print()
        centroidsList = oneIter(centroidsList, k)
    return deepcopy(centroidsList)


# one iteration in order to specific the centroids
def oneIter(centroidsList, k):
    dictonary = {}
    # init the dictionary to key:list of pixels
    for cent in range(k):
        dictonary[cent] = []
    # find the closest cent to the pixel
    for pixel in X:
        # index of closest centroid
        cloCent = find_closest_centroid(pixel, centroidsList)
        # add out pixel to the dic of the cent
        dictonary[cloCent].append(pixel)
    # enter original centroid to empty list
    for cent in range(k):
        if len(dictonary[cent]) == 0:
            dictonary[cent].append(centroidsList[cent])
    return calculateNewCentoids(dictonary)


# get pixel from the image (RGB format) and the centroids list
# return the closest centroid (By norm)
def find_closest_centroid(pixel, centroidsList):
    minDis = float("inf")
    minIndex = 0
    for i in range(len(centroidsList)):
        dis = distance(pixel, centroidsList[i])
        if dis < minDis:
            minIndex = i
            minDis = dis
    return minIndex


# calculate the distance from point x to specific centroid
def distance(x, cent):
    # distance between red of x and red of cent pow 2
    r = (x[0] - cent[0]) ** 2
    g = (x[1] - cent[1]) ** 2
    b = (x[2] - cent[2]) ** 2
    return math.sqrt(r + g + b)


# cpecification of each centroid to the avg of his pixels set
def calculateNewCentoids(dictionary):
    newCent = []
    for entry in dictionary:
        newCent.append(calculateOneNewCentroid(dictionary[entry]))
    return newCent


# get the pixels which closest to our centroid and rePlace the centroid
def calculateOneNewCentroid(pixelList):
    r = g = b = 0
    for pixel in pixelList:
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]
    i = len(pixelList)
    # NOTE: there isn't i=0 because i entered to an empty list the original centroid
    return [r / i, g / i, b / i]


def showThePhoto(newCentList):
    B = deepcopy(A)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            newPixel = find_closest_centroid(B[i][j], newCentList)
            B[i][j] = newCentList[newPixel]
    plt.imshow(B)
    plt.grid(False)
    plt.show()


# call the main
main()
