import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
from gpiozero import Robot

robot=Robot((18,17),(23,22))

# Select a region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon formed
	by the vertices. The rest of the image pixels is set to zero (black).
    """

    # Defining a blank mask
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending
	# on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def Velocidad(xmil , xmal, y_bl, y_tl, xmir, xmar, y_br, y_tr):
    #Left Lane
    xminl=xmil
    xmaxl=xmal
    y_bottoml=y_bl
    y_topl=y_tl
    #Right Lane
    xminr=xmir
    xmaxr=xmar
    y_bottomr=y_br
    y_topr=y_tr
    #Pendiente
    ml=(-y_topl+y_bottoml)/(xmaxl-xminl)
    mr=(-y_topr+y_bottomr)/(xmaxr-xminr)
    #B's
    bl=y_topl-ml*xminl
    br=y_bottomr-mr*xmaxr
    #Pendientes impresas
    #print("\nLas fórmulas de las rectas izq y der son: ")
    #print("y=",ml,"x +",bl)
    #|print("y=",mr,"x +",br,"\n")
    #Calculo de xr y xl en ya
    ya=250 
    xl=(ya-bl)/ml
    xr=(ya-br)/mr
    #Xroi
    xroi=(xr-xl)/2+xl
    #print("xroi es: ",xroi)
    A = np.matrix([[-ml, 1],[-mr, 1]])
    b = np.matrix([[bl],[br]])
    [yint,xint] = (A**-1)*b
    #print("Las coordenadas son: ",xint)


    constRap=0.7
    if (xroi < xint):
        correcion = (abs(xroi-xint)*constRap)/xroi
        rapIzq=constRap-correcion
        rapDer=constRap+correcion
        robot.value = (rapDer, rapIzq)
        
        print("RAPIDEZ:: "+str(rapIzq)+" "+str(rapDer))
        
    elif (xroi > xint):
        correcion = ((xroi-xint)*constRap)/xroi
        rapIzq=constRap+correcion
        rapDer=constRap-correcion
        robot.value = (rapDer, rapIzq)
        
        print("RAPIDEZ:: "+str(rapIzq)+" "+str(rapDer))
    else:
        rapIzq=constRap
        rapDer=constRap
        robot.forward(speed=abs(constRap))
        print("RAPIDEZ:: "+str(rapIzq)+" "+str(rapDer)) 
       
    return [rapIzq, rapDer]

#cap = cv2.VideoCapture(0, cv2.CAP_V4L)
#cap=cv2.VideoCapture("/home/udem/Escritorio/CarritoAutonomo/VC/PRUEBA1.mp4")
cap = cv2.imread('/home/albertocastro/Escritorio/Taller/dataset2/opencv_frame_0.png')
scap=100
mtx=np.genfromtxt('/home/albertocastro/Escritorio/Taller/dataset2/camera_mtx.out')
print(mtx)
dist=np.genfromtxt('/home/albertocastro/Escritorio/Taller/dataset2/dist_c.out',delimiter=",")
print(dist)
nmtx=np.genfromtxt('/home/albertocastro/Escritorio/Taller/dataset2/newcameramtx.out')
roi=np.genfromtxt('/home/albertocastro/Escritorio/Taller/dataset2/new_roi.out', dtype='i')
roi_x=roi[0]
roi_y=roi[1]
roi_w=roi[2]
roi_h=roi[3]


#2. COnvertir de BGR a RGB, luego de RGB a Gray
img_colour_rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)
#Resize image
scale_percent = scap
width = int(grey.shape[1]* scale_percent / 100)
height = int(grey.shape[0]* scale_percent / 100)
dim = (width, height)
print(dim)
grey = cv2.resize(grey, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow("Lane line detection", grey)
dist_frame=grey
undst_frame=cv2.undistort(dist_frame,mtx,dist,None,nmtx)
#grey=(undst_frame)
cropped_frame = undst_frame[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
grey=cropped_frame
#cv2.imshow("undistorted frame",grey)

# 3.- Apply Gaussian smoothing
kernel_size = (17,17)
blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
#cv2.imshow("Smoothed image", blur_grey)

# 4.- Apply Canny edge detector
low_threshold = 70
high_threshold = 100
edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
#cv2.imshow("Canny image", edges)


# 5.- Define a polygon-shape like region of interest
img_shape = grey.shape
#print("IMG SIZE "+str(img_shape))
# RoI (change the below vertices accordingly)

p1 = (1, 290)
p2 = (1, 232)
p3 = (223, 157)
p4 = (491, 163)
p5 = (639, 196)
p6 = (639, 287)
"""
#DEFINING NEW ROI
#p1=(1,400)
p1=(1,img_shape[0]*.65)
#p2=(2,185)
p2=(2,img_shape[0]*.25)
#p3=(755,105)
p3=(img_shape[1]*.5,img_shape[0]*.2)
#p4=(1052,146)
p4=(img_shape[1]*.75,img_shape[0]*.2)
#p5=(1293,260)
p5=(img_shape[1]-5,img_shape[0]*.4)
#p6=(1272,412)
p6=(img_shape[1]*.98,img_shape[0]*.5)
"""

# Create a vertices array that will be used for the roi
vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype=np.int32)


#6. Get region of interest using the just created polygon.
# This will be used together with the Hugh 

masked_edges = region_of_interest(edges, vertices)
#cv2.imshow("Canny image within Region Of Interest", masked_edges)

# 7.- Apply Hough transform for lane lines detection
rho = 0.5                     # distance resolution in pixels of the Hough grid
theta = np.pi/1000            # angular resolution in radians of the Hough grid
threshold = 80                # minimum number of votes (intersections in Hough grid cell)
min_line_len = 1              # minimum number of pixels making up a line
max_line_gap = 18              # maximum gap in pixels between connectable line segments
line_image = np.copy(img_colour_rgb)*0   # creating a blank to draw lines on
hough_lines = cv2.HoughLinesP(masked_edges,
                              rho,
				              theta,
				              threshold,
				              np.array([]),
				              minLineLength=min_line_len,
				              maxLineGap=max_line_gap)

if hough_lines is not None:
    print("number of lines:{}".format(hough_lines.shape))
else:
    hough_lines=[]
    print("NO SE DETECTA LINEA")
    robot.stop()

    
# 8.- Initialise a new image to hold the original image with the detected lines
# Resize img_colour_with_lines
scale_percent = scap
width = int(grey.shape[1] * scale_percent / 100)
height = int(grey.shape[0] * scale_percent / 100)
dim = (width, height)
img_colour_rgb = cv2.resize(img_colour_rgb, dim, interpolation = cv2.INTER_AREA)
img_g2rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
img_colour_with_lines = img_g2rgb.copy()

left_lines, left_slope, right_lines, right_slope = list(), list(), list(), list()
ymin, ymax, xmin, xmax = 0.0, 0.0, 0.0, 0.0
x_left, y_left, x_right, y_right = list(), list(), list(), list()

# Slope and standard deviation for left and right lane lines
#left_slope_mean, left_slope_std = -20.09187457328413, 3.4015553620470467
left_slope_mean, left_slope_std = -18.5, 4
#right_slope_mean, right_slope_std = 21.713840954352456, 1.7311898404656396
right_slope_mean, right_slope_std= 15.5, 4

# Loop through each detected line
for line in hough_lines:
    for x1, y1, x2, y2 in line:

        # Compute slope for current line
        slope = (y2-y1) / (x2-x1)
        
        slope_deg = np.rad2deg(np.arctan(slope))
        print("slope = "+str(slope_deg))
        
        #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 10)

        # If slope is positive, the current line belongs to the right lane line
        if (slope_deg >= (right_slope_mean - 1*right_slope_std)) and (slope_deg < (right_slope_mean + 1*right_slope_std)):
            right_lines.append(line)
            #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (255,0,0), 10)
            right_slope.append(slope)

            x_right.append(x1)
            x_right.append(x2)
            y_right.append(y1)
            y_right.append(y2)

        # Otherwise, the current line belongs to the left lane line
        elif (slope_deg >= (left_slope_mean - 1*left_slope_std)) and (slope_deg < (left_slope_mean + 1*left_slope_std)):
            left_lines.append(line)
            #cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 10)
            left_slope.append(slope)
            x_left.append(x1)
            x_left.append(x2)
            y_left.append(y1)
            y_left.append(y2)

        # Outliers lines; i.e., lines that neither belong to left nor right lane lines
        else:
            pass

#cv2.imshow("Canny image with detected lines", img_colour_with_lines)

if len(x_left) != 0 and len(y_left) != 0:
    x_min_l = np.min(x_left)
    x_max_l = np.max(x_left)
    y_min_l = np.min(y_left)
    y_max_l = np.max(y_left)
    print("LINEAS IZQUIERDAS")
    print("xminl= "+str(x_min_l))
    print("xmaxl= "+str(x_max_l))
    print("yminl= "+str(y_min_l))
    print("ymaxl= "+str(y_max_l))

    # Find the regression line for the left lane line
    left_regression_line = linear_model.LinearRegression()
    left_regression_line.fit(np.array(x_left).reshape(-1,1), y_left)

    y_bottom_l = left_regression_line.coef_*x_min_l + left_regression_line.intercept_
    y_top_l = left_regression_line.coef_*x_max_l + left_regression_line.intercept_
    print("ybotl= "+str(y_bottom_l[0]))
    print("ytopl= "+str(y_top_l[0]))
    
    

    # Draw a green line to depict the left lane line
    cv2.line(img_colour_with_lines, (x_min_l, int(y_bottom_l[0])), (x_max_l, int(y_top_l[0])), (0,255,0), 5)
else:
    robot.stop()
    
if len(x_right) != 0 and len(y_right) != 0:
    x_min_r = np.min(x_right)
    x_max_r = np.max(x_right)
    y_min_r = np.min(y_right)
    y_max_r = np.max(y_right)

    print("LINEAS DERECHAS")
    print("xminr= "+str(x_min_r))
    print("xmaxr= "+str(x_max_r))
    print("yminr= "+str(y_min_r))
    print("ymaxr= "+str(y_max_r))
# Fidn the regression line for the right lane line
    right_regression_line = linear_model.LinearRegression()
    right_regression_line.fit(np.array(x_right).reshape(-1,1), y_right)

    y_bottom_r = right_regression_line.coef_*x_min_r + right_regression_line.intercept_
    y_top_r = right_regression_line.coef_*x_max_r + right_regression_line.intercept_
    print("ybotr= "+str(y_bottom_r[0]))
    print("ytopr= "+str(y_top_r[0]))
# Draw a green line to depict the right lane line
    cv2.line(img_colour_with_lines, (x_min_r, int(y_bottom_r[0])), (x_max_r, int(y_top_r[0])), (0,255,0), 5)
else:
        robot.stop()

cv2.imshow("Canny image with detected lines", img_colour_with_lines)
Velocidad(x_min_l, x_max_l, y_bottom_l[0], y_top_l[0], x_min_r, x_max_r, y_bottom_r[0], y_top_r[0])

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
#path_to_images = "/home/udem/CarritoAutonomo/VC/"
