import cv2
import numpy as np
import collections
import math
from scipy import ndimage


FILE1 = "data/kanji-Gothic/kanji_1.png"
FILE2 = "data/kanji-Mincho/kanji_1.png"
FILE3 = "data/kanji-Mincho/kanji_2.png"

NUM_BINS = 4

def getSurfBins(img, kps):
  height, width = img.shape
  binW = math.floor(width / NUM_BINS)
  binH = math.floor(height / NUM_BINS)

  bins = np.zeros((NUM_BINS, NUM_BINS))

  for kp in kps:
    x = min(math.floor(kp.pt[0] / binH), NUM_BINS - 1)
    y = min(math.floor(kp.pt[1] / binW), NUM_BINS - 1)
    bins[x, y] += 1

  bins.flatten()
  #print bins
  return bins.flatten()


def orbFeatures(img):
  orb = cv2.ORB_create()
  kp = orb.detect(img, None)
  bins = getSurfBins(img, kp)
  return bins

def COG(img):
  height, width = img.shape
  numPoints = 0
  heightTotal = 0
  widthTotal = 0
  for h in range(height):
    for w in range(width):
      if img[h, w] == 0:
        numPoints += 1
        heightTotal += h
        widthTotal += w

  if numPoints != 0:
    widthTotal /= numPoints
    heightTotal /= numPoints
  return np.array((widthTotal, heightTotal))

def show(img):
  cv2.imshow('title', img)
  cv2.waitKey()

def PDC_diag_features(img, bw = False):
  if bw:
    im_bw = img
  else:
    (thresh, im_bw) = cv2.threshold(
              img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  scaled = cv2.resize(im_bw, (48, 48))

  all_layers = []

  # Whoah https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
  diags = [scaled[::-1,:].diagonal(i)
            for i in range(-scaled.shape[0]+1,scaled.shape[1])]
  diags.extend(scaled.diagonal(i) for i in range(scaled.shape[1]-1,-scaled.shape[0],-1))
  for run in diags:
    start = None
    layers = [0] * 3
    l_i = 0
    for i, pixel in enumerate(run):
      if start == None and pixel == 0:
        start = i
      if start != None and pixel != 0:
        layers[l_i] = i - start
        start = None
        l_i += 1
        if l_i == 3:
          break
    all_layers.append(layers)

  # not sure if this is perfect :/
  l = len(all_layers)
  results = [np.mean(all_layers[row:row+8], axis=0) for row in range(0, l, 8)]

  return np.concatenate(results).flatten()


def PDC_features(img, bw = False):
  if bw:
    im_bw = img
  else:
    (thresh, im_bw) = cv2.threshold(
              img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # black is 0, white is 255
  scaled = cv2.resize(im_bw, (48, 48))

  CARDINAL_DIRS = [
    # startcol, startrow, drow, dcol, row_order
    (0, 0, 1, 1, 1),
    (0, 0, 1, 1, 0),
    (47, 0, -1, 1, 1),
    (0, 47, 1, -1, 0)
  ]
  directionLayers = []

  for startcol, startrow, drow, dcol, row_order in CARDINAL_DIRS:
    full_row_vals = []
    for i in range(48):
      # each row
      l_i = 0
      layers = [0] * 3
      start = None
      for j in range(48):
        if row_order == 1:
          pixelVal = scaled[startcol + i * drow, startrow + j * dcol];
        else:
          pixelVal = scaled[startrow + j * dcol, startcol + i * drow];
        if start == None and pixelVal == 0:
          start = j
        if start != None and pixelVal != 0:
          layers[l_i] = abs(abs(dcol * j + drow * (i))-start)
          l_i += 1
          start = None
          if l_i == 3:
            break
      full_row_vals.append(layers)
    directionLayers.append(full_row_vals);

  results = [
      [np.mean(row_vals[i:i+8], axis=0) for i in range(0,48, 8)]
      for row_vals in directionLayers
  ]

  return np.concatenate(results, axis=1).flatten()

def global_features(img):
    height, width = img.shape
    height_width_ratio = 1. * height / width
    x_com, y_com = ndimage.measurements.center_of_mass(img)

    row_std = np.mean(np.std(img, axis=0))
    col_std = np.mean(np.std(img, axis=1))

    num_blk_pixels = height * width - (1. * np.sum(img) / 255)
    ratio_filled = num_blk_pixels / (height * width)

    return np.array([
        height_width_ratio,
        x_com,
        y_com,
        row_std,
        col_std,
        ratio_filled,
    ])

def all_features(img, bw = False):
  if not bw:
    (thresh, im_bw) = cv2.threshold(
                img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  else:
    im_bw = img
  # black is 0, white is 255
  scaled = cv2.resize(im_bw, (48, 48))
  feats = np.concatenate((
    global_features(img),
    PDC_features(scaled, True),
    PDC_diag_features(scaled, True),
    COG(scaled),
    orbFeatures(scaled),
  ))
  return feats


def main():
  img1 = cv2.imread(FILE1, cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread(FILE2, cv2.IMREAD_GRAYSCALE)
  #img3 = cv2.imread(FILE3, cv2.IMREAD_GRAYSCALE)
  #print PDC_features(img1)
  for a,b in zip(PDC_diag_features(img1), PDC_diag_features(img2)): #, PDC_features(img3)):
    print a
    print b
    #print c
    print "=="

if __name__ == "__main__":
  main()
