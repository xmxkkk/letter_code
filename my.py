import cv2 as cv

img=cv.imread("./data/test/0_q.png")
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()