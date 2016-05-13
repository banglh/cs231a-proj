import cv2
import numpy as np

def main():
  img = cv2.imread('ai.png')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  sift = cv2.SIFT()
  kp = sift.detect(gray, None)

  final = cv2.drawKeypoints(gray, kp)

  cv2.imwrite('ai_keypoints.png', final)

if __name__ == '__main__':
  main()
