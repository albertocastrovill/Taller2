import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
    
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
    
def pipeline(path_to_images, img_name):

        #1. Leer Imagen
        #img_name = "foto1.png"
        img_colour = cv2.imread(path_to_images)

        #2. Verificar que la imagen existe
        if img_colour is None:
            print('ERROR: image ', img_name, 'could not be read')
            exit()

        #2. COnvertir de BGR a RGB, luego de RGB a Gray
        img_colour_rgb = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)
        grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)
        
        
        #Resize image
        scale_percent = 100
        width = int(grey.shape[1]* scale_percent / 100)
        height = int(grey.shape[0]* scale_percent / 100)
        dim = (width, height)
        grey = cv2.resize(grey, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Lane line detection", grey)
        
        # 3.- Apply Gaussian smoothing
        kernel_size = (17,17)
        blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
        cv2.imshow("Smoothed image", blur_grey)

        # 4.- Apply Canny edge detector
        low_threshold = 70
        high_threshold = 100
        edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
        cv2.imshow("Canny image", edges)
        
        # 5.- Define a polygon-shape like region of interest
        img_shape = grey.shape

        # RoI (change the below vertices accordingly)
        p1 = (1, 290)
        p2 = (1, 232)
        p3 = (223, 157)
        p4 = (491, 163)
        p5 = (639, 196)
        p6 = (639, 287)

        # Create a vertices array that will be used for the roi
        vertices = np.array([[p1, p2, p3, p4, p5, p6]], dtype=np.int32)

        #6. Get region of interest using the just created polygon.
        # This will be used together with the Hugh 

        masked_edges = region_of_interest(edges, vertices)
        cv2.imshow("Canny image within Region Of Interest", masked_edges)
        
        cv2.waitKey(0)
        
path_to_images = "/home/albertocastro/Escritorio/Taller/dataset2/opencv_frame_0.png"

img_name = "opencv_frame_0.png"

pipeline(path_to_images, img_name)
