import pygame
import cv2

pygame.init()   #Inicializamos módulos de pygame
pygame.joystick.init()      #Iniciazamos módulo de joystick
j=pygame.joystick.Joystick(0)   #Definimos nuestro joystick
j.init()
buttons=j.get_numbuttons()
def Fotos(i):
    cam = cv2.VideoCapture(0)
    

    #cv2.namedWindow("test")
    
    while True:
        ret, frame = cam.read()
        img_name = "opencv_frame_{}.png".format(i)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(i))
        
        break
        
    cam.release()

    cv2.destroyAllWindows()
    
    
i=0
while True:
        
    print("Regresamo")
    for event in pygame.event.get():
        if event.type==pygame.JOYBUTTONDOWN:
            if event.button == 0:
                Fotos(i)
                i=i+1
    #print(i)
                
                
