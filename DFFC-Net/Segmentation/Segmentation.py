import numpy as np
import math
import pywt
import numpy as np
from math import *
import random
import os
import cv2
import matplotlib.pyplot as plt
import math
from math import *
import os
import pywt
import json

def minJuLi(A):
    l = len(A)
    maxdata = max(A)
    mindata = min(A)
    k = maxdata - mindata
    for i in range(l-1):
        a = A[i]
        b = A[i+1]
        c = abs(a-b)
        if c!=0 and c<k:
            k = c
    return k


def hist1(A):
    l = len(A)
    maxdata = max(A)
    mindata = min(A)
    # print('maxdata =', maxdata)
    # print('mindata =', mindata)
    d = (maxdata-mindata)/256
    # C is used to store the count corresponding to each value from 0 to 255
    # T is used to record the smallest value in array A, as well as the values at every interval of d.
    C = np.zeros((256), np.uint32)
    T = []
    x = mindata
    y = maxdata
    for i in range(256):
        x = x + d
        T.append(x)

    # print("T:")
    # print(T)

    # for i in range(l):
    #     a1 = A[i]
    #     for j in range(256):
    #         k = T[j]
    #         if a1<=k:
    #             C[j] = C[j] + 1
    #             break
    for i in range(l):
        a1 = A[i]
        for j in range(256):
            k1 = T[j]
            k2 = k1-d
            if a1<=k1 and a1>=k2:
                C[j] = C[j] + 1
                break

    # print(C)

    return C, T, d


def ArrayMax(array):
    # Find the index of the largest value in the array
    k = 0
    t = array[0]
    l = len(array)
    for i in range(1, l):
        a = array[i]
        if a > t:
            t = a
            k = i

    return k


def noZeroArray(h1):
    # Used to extract non-zero values from the histogram array into N1, and store their indices in N2.
    l = len(h1)
    N1 = []
    N2 = []
    for i in range(l):
        a = h1[i]
        if a > 100:
            N1.append(a)
            N2.append(i)

    return N1, N2


def maxArrayIndex(array):
    l = len(array)
    t = array[0]
    k = 0
    for i in range(1, l):
        a = array[i]
        if a > t:
            t = a
            k = i

    return k


def rightPeek(N1, N2, maxIndex):
    l = len(N1)
    M = []
    N = []
    for i in range(maxIndex, l):
        a = N1[i]
        b = N2[i]
        M.append(a)
        N.append(b)

    M = np.array(M)
    N = np.array(N)

    return M, N

def rightPeek_1(A, maxIndex):
    l = len(A)
    M = []
    N = []
    for i in range(maxIndex, l):
        a = A[i]
        b = i
        M.append(a)
        N.append(b)

    M = np.array(M)
    N = np.array(N)

    return M, N




def hanshuqiudao(a):
    l = len(a)
    Y = []
    for i in range(l, 1, -1):
        z = i - 1
        j = l - i
        # print('z =', z, ' l-i =', j)
        b = z * a[j]
        Y.append(b)
    Y = np.array(Y)

    return Y



def guaidian(a, X):
    l = len(X)
    aly = 1
    p = np.poly1d(a)
    b0 = p(X[0])
    S = []
    if b0 == 0:
        S.append(0)
    for i in range(1, l-1):
        m1 = i - aly
        m2 = i + aly
        b1 = p(X[m1])
        b2 = p(X[m2])
        if (b1 > 0 and b2 < 0) or (b1 < 0 and b2 > 0):
            S.append(i)

    m2 = l-1
    bl = p[X[m2]]
    if bl == 0:
        S.append(m2)

    S = np.array(S)


    return S



def guaiDian2(a, X):
    l = len(X)
    p = np.poly1d(a)
    s = 0
    for i in range(1, l-1):
        m1 = i - 1
        m2 = i + 1
        b1 = p(X[m1])
        b2 = p(X[m2])
        if (b1 > 0 and b2 < 0) or (b1 < 0 and b2 > 0):
            s = i
            break


    return s
def hist_temperature(A):
    l = len(A)


def halfPeekIndex(A, halfA, maxAIndex):
    l = len(A)
    left = maxAIndex
    right = maxAIndex
    for i in range(0, maxAIndex):
        a = A[i]
        if a >= halfA:
            left = i
            break

    for i in range(maxAIndex, l):
        a = A[i]
        if a <= halfA:
            right = i
            break

    return left, right


def halfPeekIndex1(A, halfA, maxAIndex):
    l = len(A)
    left = maxAIndex
    right = maxAIndex
    for i in range(0, maxAIndex):
        a = A[i]
        if a >= halfA:
            left = i
            break

    i = l - 1
    while (i > maxAIndex):
        a = A[i]
        if a >= halfA:
            right = i
            break
        i = i - 1

    return left, right


def halfPeekIndex2(A, halfA, maxAIndex):
    l = len(A)
    left = maxAIndex
    for i in range(0, maxAIndex):
        a = A[i]
        if a >= halfA:
            left = i
            break

    return left


def left_right_avg(A, left, right):
    sumA = sum(A)
    m = 0
    c = 0
    for i in range(left, right + 1):
        c = c + 1
        a = A[i]
        m = m + a
    avg = m / c

    return avg


def mindistance(A):
    # Calculate the minimum spacing between adjacent elements in A
    l = len(A)
    # print("A[1] =", A[1])
    # print("A[0] =", A[0])
    d = abs(A[1] - A[0])
    Distance = []
    Distance.append(d)
    for i in range(1, l):
        a = A[i]
        b = A[i - 1]
        temp = abs(a - b)
        Distance.append(temp)
        if (temp < d and temp != 0) or d == 0:
            d = temp

    return d, Distance


def zhebanchazhao(T, a, d):
    l = len(T)
    start = 0
    end = l - 1
    index = -1
    while start <= end:

        mid = (start + end) // 2
        Tmid = T[mid]
        if a <= Tmid and a >= (Tmid - d):
            index = mid
            break
        elif Tmid < a:
            start = mid + 1
        else:
            end = mid - 1

    return index


def hist2(A):
    # Directly use the temperature data as the horizontal axis, calculate the spacing between adjacent elements, and select the minimum distance as d.
    l = len(A)
    maxdata = max(A)
    mindata = min(A)
    # print('maxdata =', maxdata)
    # print('mindata =', mindata)
    d, distance = mindistance(A)
    d_mean = np.mean(distance)
    d = d_mean / 8

    k = int(np.ceil((maxdata - mindata) / d))

    C = np.zeros((k + 1), np.uint64)
    T = []
    i = mindata
    while (i <= maxdata):
        i = i + d
        T.append(i)
    l_T = len(T)
    # print(T)

    for i in range(l):
        a1 = A[i]
        index = zhebanchazhao(T, a1, d)
        C[index] = C[index] + 1

    return C, T, d


def hist3(A):
    l = len(A)
    maxdata = max(A)
    mindata = min(A)
    d = 0.002
    k = int(np.ceil((maxdata - mindata) / d))

    C = np.zeros((k), np.uint64)
    T = []
    i = mindata
    while (i <= maxdata):
        i = i + d
        T.append(i)

    l_T = len(T)

    for i in range(l):
        a1 = A[i]
        index = zhebanchazhao(T, a1, d)
        C[index] = C[index] + 1

    return C, T, d


def hist4(A):

    l = len(A)
    maxdata = max(A)
    mindata = min(A)
    k = 10000
    d = int(np.ceil((maxdata - mindata) / k))

    C = np.zeros((k), np.uint64)
    T = []
    i = mindata
    while (i <= maxdata):
        i = i + d
        T.append(i)
    l_T = len(T)
    for i in range(l):
        a1 = A[i]
        index = zhebanchazhao(T, a1, d)
        C[index] = C[index] + 1

    return C, T, d


def threshImage(image, t):
    tImage = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > t:
                tImage[i, j] = 255

    return tImage



def T_xuanze(imgdata):
    image = imgdata.copy()
    img1D = np.array(image).flatten()
    img1D = img1D.tolist()
    imagedata = img1D.copy()

    hist, XX, d = hist1(img1D)
    maxhist = max(hist)
    maxhistIndex = maxArrayIndex(hist)
    halfmaxhist = maxhist / 2
    left, right = halfPeekIndex(hist, halfmaxhist, maxhistIndex)
    u1 = XX[maxhistIndex]
    tt = 2 * sqrt(2 * log(2, e))
    a1 = (XX[right] - XX[left]) / tt

    ua1 = u1 + a1

    T = ua1
    return T

def dwt(img, T):
    img_1D = np.array(img).flatten()
    q = sqrt(2*(math.log10(256*640)))*T
    q_0 = T
    #q_0 = 0.25*q_0
    q_0 = 0.04*q_0
    data_soft = pywt.threshold(data=img_1D, value=q_0, mode='soft', substitute=q_0)
    img_dwt = data_soft.reshape([256, 640])
    return img_dwt



def threshold_mean(img, k):
    a, b = img.shape
    mean_image = np.zeros(img.shape, np.float)
    for i in range(a):
        for j in range(b):
            ksize = KsizeVl(k)
            # print(img[i, j])
            pixvalue = pixValue(img, ksize, i, j, k)
            # print(pixvalue)
            # f = countPixT(img, t, k, i, j)
            mean_image[i, j] = pixvalue
            # if pixvalue >= t:
            #     img[i, j] = 255
            # else:
            #     img[i, j] = 0
    return mean_image


def pixValue(img, ksize, i, j, k):
    c = math.floor(k/2)
    sum_pix = 0
    h = img.shape[0]
    w = img.shape[1]
    m = 0
    count = 0
    for x in range(i-c, h):
        n = 0
        if x>=0 and m<k:
            for y in range(j-c, w):
                if y>=0 and n<k:
                    # print('m:', m, ',n:', n, ',x:', x, ',y:', y)
                    # print('ksize[m, n]:', ksize[m, n])
                    # print('img[x, y]:', img[x, y])
                    # print('ksize[m, n]*img[x, y]:',ksize[m, n]*img[x, y])
                    sum_pix = sum_pix + ksize[m, n]*img[x, y]
                    count = count + 1
                    y = y + 1
                    n = n + 1
                else:
                    y = y + 1
                    n = n + 1
                if n >= k:
                    x = x + 1
                    m = m + 1
                    break
        else:
            x = x + 1
            m = m + 1
        if m>=k:
            break
        else:
            continue

    sumKsize = np.sum(ksize)
    pixvalue = sum_pix/sumKsize
    return pixvalue


def KsizeVl(k):
    ksize = np.ones([k, k])
    a = random.random()
    # The larger the value of a, the weaker the denoising ability, and the original pixel values of the image play a greater decisive role.
    while (a < 0.6 or a > 0.63):
        a = random.random()
        # print('while:', a)
    # print(a)
    i = math.floor(k / 2)
    ksize[i, i] = ksize[i, i] * a
    i = i - 1
    c = 1
    # print(c, ':', a)
    while (i >= 0):
        if i > 0:
            a = a / 2
        else:
            j = math.floor(k / 2)
            a = 1
            while (j > 0):
                a = a - ksize[j, j]
                j = j - 1
        c = c + 2
        # print(c, ':', a)
        count = c * 2 + (c - 2) * 2
        t = a / count
        for x in range(i, i + c):
            ksize[i, x] = t
            ksize[i + c - 1, x] = t
            x = x + 1
        for y in range(i + 1, i + c - 1):
            ksize[y, i] = t
            ksize[y, i + c - 1] = t
            y = y + 1
        i = i - 1

    # print(ksize)
    return ksize
def T_xuanze_Tl_Tr(imgdata):
    image = imgdata.copy()
    img1D = np.array(image).flatten()
    img1D = img1D.tolist()
    imagedata = img1D.copy()

    hist, XX, d = hist2(img1D)
    maxhist = max(hist)
    maxhistIndex = maxArrayIndex(hist)
    halfmaxhist = maxhist / 2
    left, right = halfPeekIndex(hist, halfmaxhist, maxhistIndex)
    Tm = XX[maxhistIndex]
    Tl = XX[left]
    Tr = XX[right]
    aa = sqrt(2 * log(2, e))
    al = (Tm - Tl) / aa
    ar = (Tr - Tm) / aa
    a = al
    if ar < al:
        a = ar

    N = 2
    k = sqrt(2 * log2(N))
    T = Tm + k * a

    return T


def T_xuanze_Tl_Tr_d(imgdata):
    image = imgdata.copy()
    img1D = np.array(image).flatten()
    img1D = img1D.tolist()
    imagedata = img1D.copy()

    hist, XX, d = hist3(img1D)
    maxhist = max(hist)
    maxhistIndex = maxArrayIndex(hist)
    halfmaxhist = maxhist / 2
    left, right = halfPeekIndex(hist, halfmaxhist, maxhistIndex)
    Tm = XX[maxhistIndex]
    Tl = XX[left]
    Tr = XX[right]
    # aa = sqrt(2 * log(2, e))
    aa = 1.775
    al = (Tm - Tl) / aa
    ar = (Tr - Tm) / aa
    a = al
    if ar < al:
        a = ar

    # N = 2
    # k = sqrt(2 * log2(N))

    N = 256*640
    k = sqrt(2 * log10(N))
    T = Tm + k * a

    return T

def writeToJson(path, data):

    with open(path, 'w+') as file_obj:
        json.dump(data, file_obj)


dir_1 = r'../../data_BP/'
# The root path for saving the binarized result as an image
dir_2 = 'image'
# The root path for saving the binarized results as npy format data
dir_3 = 'npy'

save_tt = 'all.json'



dirfiles = os.listdir(dir_1)


frame = 24

TTall = []

for dirfilename in dirfiles:
    openfile_dir = dir_1 + dirfilename + '/'
    files = os.listdir(openfile_dir)
    for filename in files:
        print("filename=", filename)
        tt11 = dict()
        ff = filename[:-4]

        openPath = openfile_dir + filename
        data = np.load(openPath)
        print("data_loading....！")
        h,w,frames = data.shape
        #First, average multiple frames, and then perform threshold segmentation.
        img = np.zeros_like(data[:, :, 0])
        for frame in range(20, frames-14):
            img = img + data[:, :, frame]
        img = img /(frames-34)

        image_yuantu = img.copy()
        image = img.copy()
        imgData = img.copy()

        T = T_xuanze_Tl_Tr_d(imgData)*2.3
        print("T =", T)

        tt11[ff] = T

        TTall.append(tt11)

        # Gaussian filter
        image = cv2.GaussianBlur(image, (3, 3), 1)
        image = threshold_mean(image, 3)
        image = dwt(image, T)
        image = threshImage(image, T)

        # Corrosion operation
        image = cv2.erode(image, (3, 3))
        image_name = filename[:-4]
        image_path = image_name + '.jpg'
        image_path3 = image_name + '.npy'
        save_path = dir_2 + '/' + image_path
        save_path3 = dir_3 + '/' + image_path3
        cv2.imwrite(save_path, image)
        np.save(save_path3, image)

writeToJson(save_tt, TTall)