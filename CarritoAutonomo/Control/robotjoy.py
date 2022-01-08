import pygame                                                   #Importar librería
from gpiozero import Robot                                      #Importar módulo Robot de librería gpiozero

robot=Robot((17,18),(22,23))                                    #Definir que un motor del robot se conecta a GPIO 17 y 18 y el otro motor a GPIO 22 y 23

pygame.init()                                                   #Inicializamos módulos de pygame
pygame.joystick.init()                                          #Iniciazamos módulo de joystick

j=pygame.joystick.Joystick(0)                                   #Definimos nuestro joystick
j.init()                                                        #Inicializamos nuestro joystick
print("Joystick iniciado")                                      #Imprimimos para asegurarnos que hubo conexión

while True:                                                     #Loop eterno
    pygame.event.pump()                                         #Se encarga de eventos internos
    
    x=j.get_axis(0)/.8                                          #Obtenemos la posición actual de eje x de joystick izquierdo (estandarizando de acuerdo a pruebas de caracterización)
    if x>1:                                                     #Asegurarse que la estandarización no generó valores fuera del rango -1 a 1
        x=1
    elif x<-1:
        x=-1
    y=j.get_axis(0)/.8                                          #Lo mismo para la posición de eje y de joystick derecho
    if y>1:
        y=1
    elif y<-1:
        y=-1
    
    if y<-0.1:                                                  #Si se detectó movimiento hacia adelante (colchón por caracterización)
        if x>0.1:                                               #Si se detectó movimiento hacia la derecha (colchón)
            robot.forward(speed=abs(y),curve_right=x)           #Mover el robot hacia adelante, a la velocidad que indique y y con la curva a la derecha que indique x
            print("Forward right")                              #Imprimir para etapa de pruebas
        elif x<-0.1: #curve left                                #Si el movimiento era a la izquierda
            robot.forward(speed=abs(y),curve_left=abs(x))       #Cambiar el tipo de curva
            print("Forward left")
        else:                                                   #Si no se detectó movimiento en x
            robot.forward(speed=abs(y))                         #Mover al robot sin curva
            print("Forward")
    elif y>0.1: #backward                                       #Lo mismo, pero para la reversa
        if x>0.1: #curve right
            robot.backward(speed=y,curve_right=x)
            print("Backward right")
        elif x<-0.1: #curve left
            robot.backward(speed=y,curve_left=abs(x))
            print("Backward left")
        else:
            robot.backward(speed=y)
            print("Backward")
    else:                                                       #Si no se detectó movimiento en y, detener al robot
        robot.stop()
        print("Stop")