from gpiozero import Motor  #importar módulo de motores de libreria gpiozero
from time import sleep      #importar función sleep de libreria time

motor=Motor(17,18)          #Definimos que nuestro motor está conectado a GPIO17 y GPIO18

while True:                 #Loop infinito
    motor.forward()         #Mover el motor en una dirección
    sleep(5)                #Esperar 5 segundos
    motor.backward()        #Mover el motor en la dirección contraria
    sleep(5)                #Esperar 5 segundos