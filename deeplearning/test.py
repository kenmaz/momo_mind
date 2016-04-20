import numpy as np
import cv2

def kanako():
  img = cv2.imread('img/kanako1.png')
  print img

  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def write_black():
  img = np.zeros((100,100,3), np.int8)
  cv2.imshow('result', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite('result.png', img)

img = np.zeros((255,255,3), np.uint8)
for y in range(255):
  for x in range(255):
    img[y][x] = np.array([254,0,0])
print img
cv2.imwrite('result.png', img)
#cv2.imshow('result', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
