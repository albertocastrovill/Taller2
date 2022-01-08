import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 45, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images=['chess1.jpg','chess2.jpg']
images=glob.glob('/home/albertocastro/Escritorio/Taller/calib/*.png') #DIRECCION DE TUS IMAGENES
print(images)
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('/home/albertocastro/Escritorio/Taller/calib/chess1.png')

h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(newcameramtx)
np.savetxt('camera_mtx.out',mtx,delimiter=" ")
np.savetxt('dist_c.out',dist,delimiter=",")
np.savetxt("newcameramtx.out",newcameramtx,delimiter=" ")
np.savetxt('new_roi.out',roi,delimiter=",")
print(dist)
val=0
val=input("SELECCIONE METODO 1 o 2::: ")
if (val==1):
    #METODO 1
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    
    dst = dst[y:y+h, x:x+w]
    print(dst)
    cv.imwrite('calibrate/Carrito/calibresult1.png', dst)
    print("SE GUARDO IMAGEN CALIBRADA")
elif (val==2):
    #METODO 2
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    print(str(x)+" "+str(y)+" "+str(w)+" "+str(h))
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('/home/albertocastro/Escritorio/Taller/calib/calibresult2.png', dst)
else:
    pass

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
