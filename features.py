import cv2
import numpy as np
import collections



FILE1 = "data/kanji-Gothic/kanji_1.png"
FILE2 = "data/kanji-Mincho/kanji_1.png"
FILE3 = "data/kanji-Mincho/kanji_2.png"

def show(img):
  cv2.imshow('title', img)
  cv2.waitKey()

def PDC_features(img, bw = False):

  # black is 0, white is 255
  if bw:
    im_bw = img
  else:
    (thresh, im_bw) = cv2.threshold(
              img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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


def main():
  img1 = cv2.imread(FILE1, cv2.IMREAD_GRAYSCALE)
  #img2 = cv2.imread(FILE2, cv2.IMREAD_GRAYSCALE)
  #img3 = cv2.imread(FILE3, cv2.IMREAD_GRAYSCALE)
  print PDC_features(img1)
  #for a,b,c in zip(PDC_features(img1), PDC_features(img2), PDC_features(img3)):
    #print a
    #print b
    #print c
    #print "=="

if __name__ == "__main__":
  main()
