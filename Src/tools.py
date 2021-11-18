# -*- coding: utf-8 -*-
# !/usr/bin/python

# Importações

import math
import os
import cv2 as cv
import numpy as np
from matplotlib import image, pyplot as plt
from skimage import color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from PIL import Image


# Funções


def showIm(img, name="Image"):

    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows


def preProcessing(img):
    img1 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img2 = cv.Sobel(img1,
                    cv.CV_8U,
                    1,
                    1,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv.BORDER_CONSTANT)

    img3 = img1 + img2
    img4 = autoCanny(img, 1)
    img5 = img2 + img4
    img6 = cv.threshold(
        img1, 0, 255, type=cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # img2 = cv.medianBlur(img1, 51)
    # img5 = cv.equalizeHist(img2)
    # img6 = cv.threshold(img3, 25, 255, cv.THRESH_BINARY)[1]

    titles = ['Original Image', 'Gray Image', 'Sobel Lines',
              'Sobel + Gray', 'Canny', 'Canny + Sobel', 'Otsu']
    images = [img, img1, img2, img3, img4, img5, img6]

    for i in range(7):
        plt.subplot(1, 7, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return img5


def autoCanny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(img, lower, upper)
    # return the edged image
    return edged


def extractObject(img):

    lower = np.array([140, 140, 140])
    upper = np.array([255, 255, 255])

    mask = cv.inRange(img, lower, upper)
    # showIm('mask', mask)

    mask = cv.bitwise_not(mask)
    img = cv.bitwise_and(img, img, mask=mask)

    return img


def centralPixelExtract(img):

    dim = np.shape(img)

    xDim = math.ceil(dim[0]/2)
    yDim = math.ceil(dim[1]/2)

    h = img[xDim, yDim, 0]
    s = img[xDim, yDim, 1]
    v = img[xDim, yDim, 2]

    lower = np.array([h-5, s-60, v-128])
    upper = np.array([h+4, s+50, v+128])

    mask = cv.inRange(img, lower, upper)
    img = cv.bitwise_and(img, img, mask=mask)

    return img


def createArchiveList(file):
    files = []
    for (diretorio, subpastas, arquivos) in os.walk(file):
        for arquivo in arquivos:
            files.append(os.path.join(diretorio, arquivo))

    return files


def genGLCM(img):
    gray = color.rgb2gray(img)
    img1 = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144,
                     160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(img1, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds,
                                      [1],
                                      [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                      levels=max_value,
                                      normed=False,
                                      symmetric=False)

    results = (greycoprops(matrix_coocurrence, 'contrast'),
               greycoprops(matrix_coocurrence, 'dissimilarity'),
               greycoprops(matrix_coocurrence, 'homogeneity'),
               greycoprops(matrix_coocurrence, 'energy'),
               greycoprops(matrix_coocurrence, 'correlation'),
               greycoprops(matrix_coocurrence, 'ASM'))

    return results


def houghTransform(img):
    # Load picture, convert to grayscale and detect edges
    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    """ image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                     low_threshold=0.55, high_threshold=0.8) """
    edges = autoCanny(image_rgb)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    """ result = hough_ellipse(edges, accuracy=20, threshold=250,
                           min_size=100, max_size=120) """
    result = hough_ellipse(edges)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2,
                                    nrows=1,
                                    figsize=(8, 4),
                                    sharex=True,
                                    sharey=True,
                                    # subplot_kw={'adjustable': 'box-forced'}
                                    )

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()


def windownize(path):
    img = cv.imread(path)
    x, y, _ = np.shape(img)
    centerX, centerY = round(x/2), (y/2)

    top = centerX-280
    bottom = centerX+140
    right = centerY+170
    left = centerY-130

    img1 = Image.open(path)
    img2 = img1.crop((left, top, right, bottom))

    img3 = np.array(img2)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

    return img3


if __name__ == '__main__':
    print(np.__name__)
    print(plt.__name__)
