import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten

import collections

EXAMPLE = "data/matthew/matthew_1.png"

def split(im_name=EXAMPLE):
  img = cv2.imread(EXAMPLE, cv2.IMREAD_GRAYSCALE) 
  (thresh, im_bw) = cv2.threshold(
      img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  height,width = im_bw.shape

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

  MIN_WIDTH = 10
  chars = collections.defaultdict(list)
  for li, (row, (rstart, rend)) in enumerate(lines):
    start = None
    for i in range(width):
      col = row[:,i]
      if start == None and sum(col) == 0:
        continue
      elif start == None:
        start = i
      elif start != None and sum(col) == 0 and i - start > MIN_WIDTH:
        chars[li].append((img[rstart:rend,:][:, start:i], (start, i)))
        start = None
  
  #return a flattened list
  return chars

  #for li, (row, (rstart, rend)) in enumerate(lines):
  #  for _, (cstart, cend) in chars.get(li):
  #    cv2.rectangle(img, (cstart, rstart), (cend, rend), 0)
  
  #cv2.imshow('asdf', img)
  #cv2.waitKey()
  
if __name__ == '__main__':
  main()
