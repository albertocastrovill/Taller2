import pygame                                                   #Importar librería
from gpiozero import Robot                                      #Importar módulo Robot de librería gpiozero

robot=Robot((17,18),(23,22))                                    #Definir que un motor del robot se conecta a GPIO 17 y 18 y el otro motor a GPIO 22 y 23

pygame.init()                                                   #Inicializamos módulos de pygame
pygame.joystick.init()                                          #Iniciazamos módulo de joystick

j=pygame.joystick.Joystick(0)                                   #Definimos nuestro joystick
j.init()                                                        #Inicializamos nuestro joystick
print("Joystick iniciado")                                      #Imprimimos para asegurarnos que hubo conexión

while True:                                                     #Loop eterno
    pygame.event.pump()                                         #Se encarga de eventos internos
    
    x=j.get_axis(1)/.8                                          #Obtenemos la posición actual de eje x de joystick izquierdo (estandarizando de acuerdo a pruebas de caracterización)
    if x>1:                                                     #Asegurarse que la estandarización no generó valores fuera del rango -1 a 1
        x=1
    elif x<-1:
        x=-1
    #elif (x>-0.01 and x<0.01):
     #   x=0
    y=j.get_axis(2)/.8                                       #Lo mismo para la posición de eje y de joystick derecho
    if y>1:
        y=1
    elif y<-1:
        y=-1
    #elif (y>-0.01 and y<0.01):
    #    y=0
    
    
    if x<-0.1:                                                  #Si se detectó movimiento hacia adelante (colchón por caracterización)
        if y>0.1:                                               #Si se detectó movimiento hacia la derecha (colchón)
            robot.forward(speed=abs(x),curve_right=y)      #Mover el robot hacia adelante, a la velocidad que indique y y con la curva a la derecha que indique x
            print("Forward right")                              #Imprimir para etapa de pruebas
        elif y<-0.1: #curve left                                #Si el movimiento era a la izquierda
            robot.forward(speed=abs(x),curve_left=abs(y))       #Cambiar el tipo de curva
            print("Forward left")
        else:                                                   #Si no se detectó movimiento en x
            robot.forward(speed=abs(x))                         #Mover al robot sin curva
            print("Forward")
    elif x>0.1: #backward                                       #Lo mismo, pero para la reversa
        if y>0.1: #curve right
            robot.backward(speed=x,curve_right=y)
            print("Backward right")
        elif y<-0.1: #curve left
            robot.backward(speed=x,curve_left=abs(y))
            print("Backward left")
        else:
            robot.backward(speed=x)
            print("Backward")
    else:                                                       #Si no se detectó movimiento en y, detener al robot
        robot.stop()
        print("Stop")
