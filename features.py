import cv2
import numpy as np
import collections



FILE = "ai-trimmed.png"

def show(img):
  cv2.imshow('title', img)
  cv2.waitKey()
  

def main():
  img = cv2.imread(FILE, cv2.IMREAD_GRAYSCALE)

  # black is 0, white is 255
  (thresh, im_bw) = cv2.threshold(
            img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  scaled = cv2.resize(im_bw, (48, 48))


  CARDINAL_DIRS = [
    # startcol, startrow, drow, dcol
    (0, 0, 1, 0)
  ]
  full_row_vals = []
  for i in range(48):
    # each row
    l_i = 0
    layers = [0] * 3
    start = None
    for j in range(48):
      if start == None and scaled[i,j] == 0:
        start = j
      if start != None and scaled[i,j] != 0:
        layers[l_i] = j-start
        l_i += 1
        start = None
        if l_i == 3:
          break
    full_row_vals.append(layers)

  print full_row_vals

  vals = [np.mean(full_row_vals[i:i+8], axis=0) for i in range(0,48, 8)]
  print vals




main()
