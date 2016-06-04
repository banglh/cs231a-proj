import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

import collections

EXAMPLE = "data/matthew/matthew_0.png"
STOPPING_WIDTH = 50

def estimate_pitch(row):
  height, width = row.shape
  start = 0
  white = True
  widths = []
  for i in range(width):
    col = row[:,i]
    if start == 0 and sum(col) == 0:
      continue
    elif start == 0:
      start = i
      white = False
    elif sum(col) == 0:
      white = True
    elif white and sum(col) != 0:
      white = False
      widths.append(i - start)
      start = i

  widths.append(width - start)
      

  widths = np.array(widths)

  avg = np.average(widths)
  std = np.std(widths)

  widths = widths[abs(widths - avg) <= std]
  avg = np.average(widths)

  return int(np.asscalar(avg))

def trim_char(im):
  height, width = im.shape
  start = 0
  end = width - 1
  for i in range(width):
    col = im[:,i]
    if sum(col) == 0:
      start = i
    else:
      break

  for i in range(width):
    col = im[:,width - 1 - i]
    if sum(col) == 0:
      end = width - 1 - i
    else:
      break

  return im[:, start:end]

def trim_line(row):
  height, width = row.shape
  start = 0
  end = width - 1

  result = None

  white = True
  for i in range(width):
    col = row[:,i]
    if start == 0 and sum(col) == 0:
      continue
    elif start == 0:
      start = i
      white = False
    elif sum(col) == 0 and not white:
      white = True
      end = i
    elif i >= width - 1:
      end = min(width - 1, end + 3)
      start = max(0, start - 3)
      #if result == None:
      result = row[:, start:end]
      #else:
      #  result = np.hstack((result, row[:, start:end]))
      break
      start = 0
      white = True
      end = width - 1
    elif sum(col) != 0:
      white = False

  return result

def split_line(row):
  row = trim_line(row)
  pitch = estimate_pitch(row)
  height, width = row.shape
  chars = []
  r = 0
  while r < width:
    minBlack = float("inf")
    bestPitch = pitch
    pitches = []
    for i in range(int(pitch * .8) , int(pitch * 1.2)):
      if r + i >= width:
        break
      black = sum(row[:, r+i])

      if black < minBlack:
        bestPitch = i
        minBlack = black

    if minBlack == float("inf"):
      break

    contiguous = False
    for i in range(int(pitch * .8) , int(pitch * 1.2)):
      if r + i >= width:
        break
      black = sum(row[:, r+i])

      if black <= minBlack + 2:
        contiguous = True
        pitches.append(i)
      elif contiguous:
        break

    if len(pitches) < 3:
      bestPitch = int(np.asscalar(np.average(pitches)))
    else:
      bestPitch = max(int(np.asscalar(np.average(pitches))), pitches[len(pitches) - 2])

    if r + bestPitch >= width:
      break

    orig = row[:, r:r+bestPitch]
    trimmed = trim_char(orig)

    theight, twidth = trimmed.shape

    if twidth > 0:
      chars.append(orig)
      r += bestPitch
    else:
      r += pitch

  #for r in range(bestOffset, width, bestPitch):
   # if r + pitch <= width:
    #  cv2.imshow('asdf', row[:, r:r+pitch])
     # cv2.waitKey()
      #chars.append(row[:, r:r+pitch])

  return chars

def findAngle(im):
  maxValley = 0
  angle = 0
  for i in range(-45,45):
    rows,cols = im.shape 
    M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
    dst = cv2.warpAffine(im,M,(cols,rows))
    rowsums = np.sum(dst, axis=1)
    zeros = (rowsums < 5).sum()
    if zeros > maxValley:
      maxValley = zeros
      angle = i

  return angle

def findLines(im_bw):
  rowsums = np.sum(im_bw, axis=1)

  lines = []
  start = None
  for i, r in enumerate(rowsums[1:]):
    if start == None and r < 5:
      continue
    elif start == None:
      start = i
    elif start != None and r < 5:
      # we are done with this row
      lines.append((im_bw[start:i], (start, i)))
      start = None

  return lines

def split(im_name=EXAMPLE):
  img = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE) 
  (thresh, im_bw) = cv2.threshold(
      img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  angle =  findAngle(im_bw)

  height,width = im_bw.shape

  M = cv2.getRotationMatrix2D((width/2, height/2), angle,1)
  im_bw = cv2.warpAffine(im_bw, M, (width, height))

  lines = findLines(im_bw)

  chars = collections.defaultdict(list)
  for li, (row, (rstart, rend)) in enumerate(lines):
    chars[li] = split_line(row)
  
  #return a flattened list
  return chars

  #for li, (row, (rstart, rend)) in enumerate(lines):
  #  for _, (cstart, cend) in chars.get(li):
  #    cv2.rectangle(img, (cstart, rstart), (cend, rend), 0)
  
  #cv2.imshow('asdf', img)
  #cv2.waitKey()
  
if __name__ == '__main__':
  split()
