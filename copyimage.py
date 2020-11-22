import copy
import cv2

original_img = cv2.imread("./1.jpg")
img_clone = img_src.copy()
cv2.imwrite('i.jpg',img_clone)