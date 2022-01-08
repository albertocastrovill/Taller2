# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



#Get a region of interest
def region_of_interest(img, vertices):
    #mask with 0
    mask = np.copy(img)*0
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


img_name = "/home/albertocastro/Escritorio/Taller/image-dataset/video1.mp4"

param={'rho':1, 'theta':np.pi/180, 'threshold':15, 'min_line_len':5,'max_line_gap':150,
            'bottom_left':(400, 590),'top_left':(565, 380),'top_right':(630, 380),'bottom_right':(940, 590),
            'min':400,'max':590,}
    
cap = cv2.VideoCapture(img_name)


while(cap.isOpened()):

    ret, frame = cap.read()
    #BRG TO GRAY why did i use RGB2GRAY? neta neta why?
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    ###TO-DO ... BETTER RESULTS
    ##############################
    

    ###############################

    #Apply Canny edge detector
    low_threshold = 10
    high_threshold = 70
    edges = cv2.Canny(grey, low_threshold, high_threshold, apertureSize=3)

    #ROI Data
    bottom_left = param['bottom_left']
    top_left = param['top_left']
    top_right = param['top_right']
    bottom_right = param['bottom_right']

    # ROI vertice
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)

    ## TO DO Region of Interest
    ###########################################################
    
    
    ##########################################################

    # 7.- Apply Hough transform for lane lines detection
    rho = param['rho']                     # distance resolution in pixels of the Hough grid
    theta = param['theta']            # angular resolution in radians of the Hough grid
    threshold = param['threshold']                # minimum number of votes (intersections in Hough grid cell)
    min_line_len = param['min_line_len']              # minimum number of pixels making up a line
    max_line_gap = param['max_line_gap']              # maximum gap in pixels between connectable line segments
    
    
    
    hough_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    if hough_lines is not None:
        #lines arrays
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        #get slope
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) #slope
                if math.fabs(slope) < 0.4: #Only consider extreme slope
                    continue  
                if slope <= 0: #Negative slope, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else: #Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
    

        if len(left_line_x)>0 and len(left_line_y)>0 and len(right_line_x)>0 and len(right_line_y)>0:
        
            #min and max of the line
            min_y = param['min'] 
            max_y = param['max']
            
            
            #Create a function that match with all the detected lines
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))
            
            #get the start and the end
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
            
            #Create a function that match with all the detected lines
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))
            
            #get the start and the end
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
    
            #save points
            define_lines=[[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]]
        
            #Add hough lines in the image images
            img_colour_with_lines = frame.copy()
            for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,255,0), 5)
                

            #Add both lines
            img_colour_with_Definelines = frame.copy()
            for line in define_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_colour_with_Definelines, (x1, y1), (x2, y2), (255,0,0), 12)
            
            
            cv2.namedWindow('LINES', cv2.WINDOW_NORMAL)
            cv2.imshow('LINES', img_colour_with_Definelines)
            cv2.resizeWindow('LINES', 1000,900)

            cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
            cv2.imshow('ROI', edges)
            cv2.resizeWindow('ROI', 1000,900)


            cv2.namedWindow('HOUGH', cv2.WINDOW_NORMAL)
            cv2.imshow('HOUGH', img_colour_with_lines)
            cv2.resizeWindow('HOUGH', 1000,900)


            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
        

