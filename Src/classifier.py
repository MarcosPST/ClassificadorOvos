# -*- coding: utf-8 -*-
# !/usr/bin/python

# Importações

from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
import numpy as np
import os
import cv2 as cv

# Caminhos

ImagesPath = 'Data/Images'
crackedImgs = 'Data/Images/Cracked-BGex'
originalsFilePath = 'Data/Images/Originals-BGex'

# Funções


def createArchiveList(file):
    files = []
    for (diretorio, subpastas, arquivos) in os.walk(file):
        for arquivo in arquivos:
            files.append(os.path.join(diretorio, arquivo))

    return files


def autoCanny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(img, lower, upper)
    # return the edged image
    return edged


def preProcessing(img):
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img2 = cv.threshold(
        img1, 0, 255, type=cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(img2, kernel, iterations=1)

    img2 = img1 * erosion

    img3 = cv.Sobel(img1,
                    cv.CV_8U,
                    1,
                    1,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv.BORDER_CONSTANT)

    img4 = cv.Canny(img, 35, 255)
    img5 = img3+img4

    return img5


def genGLCM(img):
    img1 = img_as_ubyte(img)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144,
                     160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(img1, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds,
                                      [1],  # Definição do parâmetro D da GLCM
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


def calcAvgPix(img):
    x, y, _ = np.shape(img)
    count = 0
    countH = 0
    countS = 0
    countV = 0

    for lin in range(x):
        for col in range(y):
            countH = countH + img[lin, col][0]
            countS = countS + img[lin, col][1]
            countV = countV + img[lin, col][2]
            count = count + 1

    return [round(countH/count), round(countS/count), round(countV/count)]


def colorClassKNN(files):
    imagesPath = 'Data/Images/Color Sorted'
    resultsPath = 'Data/Images/ColorSortResults'
    trainingFiles = createArchiveList(imagesPath)
    model = KNeighborsClassifier(n_neighbors=1)  # Parâmetro K do KNMM
    labels = []
    features = []

    for file in trainingFiles:
        img = cv.imread(file)
        bgr = calcAvgPix(img)
        features.append(bgr)
        label = int(file.split('_c')[-1][:-4])
        labels.append(label)

    # Train the model using the training sets
    model.fit(features, labels)

    # Sort egg by color

    for file in files:
        img = cv.imread(file)
        bgr = calcAvgPix(img)
        label = model.predict([bgr])
        tag = str(label[0])

        cv.imwrite(resultsPath + '/c' + tag + '/Sorted_' + file[37:], img)


def damageClassKNN(files):
    imagesPath = 'Data/Images/Damage Sorted'
    resultsPath = 'Data/Images/DamageSortedResults'
    trainingFiles = createArchiveList(imagesPath)
    model = KNeighborsClassifier(n_neighbors=2)
    labels = []
    features = []

    for file in trainingFiles:
        img = cv.imread(file)
        glcm = []
        img = preProcessing(img)
        mtx = genGLCM(img)

        for i in [1, 2, 4]:
            glcm.append(mtx[i][0][0])
            glcm.append(mtx[i][0][1])
            glcm.append(mtx[i][0][2])
            glcm.append(mtx[i][0][3])

        features.append(glcm)
        label = int(file.split('_c')[-1][:-4])
        labels.append(label)

    model.fit(features, labels)

    # Sort egg by damage

    for file in files:
        img = cv.imread(file)
        glcm = []
        img1 = preProcessing(img)
        mtx = genGLCM(img1)
        for i in [1, 2, 4]:
            glcm.append(mtx[i][0][0])
            glcm.append(mtx[i][0][1])
            glcm.append(mtx[i][0][2])
            glcm.append(mtx[i][0][3])

        label = model.predict([glcm])
        tag = str(label[0])

        cv.imwrite(resultsPath + '/c' + tag + '/Sorted_' + file[37:], img)


# src

filesAll = createArchiveList(ImagesPath + '/Originals + Cracked_BGex')
colorClassKNN(filesAll)
damageClassKNN(filesAll)
