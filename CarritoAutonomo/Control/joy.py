import pygame                                           #Importar librería pygame

pygame.init()                                           #Inicializar módulos
pygame.joystick.init()                                  #Inicializar módulo de joystick

print(range(pygame.joystick.get_count()))               #Imprimir en pantalla cuántos joysticks se encontraron

j=pygame.joystick.Joystick(0)                           #Definir nuestro joystick como j
j.init()                                                #Inicializar nuestro joystick
print("Joystick iniciado")                              #Mensaje para saber que sí se detectó e inicializó el joystick

name=j.get_name()                                       #Obtener nombre de control
print("Name:",name)                                     #Imprimirlo
axes=j.get_numaxes()                                    #Obtener cantidad de ejes
print("Axes:",axes)                                     #Imprimirlo
for i in range(axes):                                   #Para cada eje
    axis=j.get_axis(i)                                  #Obtener valor base (cuando no se está moviendo)
    print("Axis",i,":",axis)                            #Imprimirlo
buttons=j.get_numbuttons()                              #Obtener cantidad de botones
print("Buttons:",buttons)                               #Imprimirlo
for i in range(buttons):                                #Para cada botón
    button=j.get_button(i)                              #Obtener valor base (cuando no se están presionando)
    print("Button",i,":",button)                        #Imprimirlo
hats=j.get_numhats()                                    #Obtener cantidad de d-pads
print("Hats:",hats)                                     #Imprimirlo
for i in range(hats):                                   #Para cada d-pad
    hat=j.get_hat(i)                                    #Obtener valor pase (cuando no se está presionando)
    print("Hat",i,":",hat)                              #Imprimirlo

oldVal=[0,0,0,0]                                        #Inicializar una variable donde se almacenará el valor de cada eje
while True:                                             #Loop eterno
    pygame.event.pump()                                 #Procesa internamente handlers de eventos
    
    for i in range(axes):                               #Para cada eje
        newVal=j.get_axis(i)                            #Obtiene valor actual
        if(newVal!=oldVal[i]):                          #Si el valor actual es distinto al anterior
            print("axis",i,"=",newVal)                  #Imprimir el nuevo valor
            oldVal[i]=newVal                            #Definir valor antiguo como el actual
            
    event=pygame.event.get()                            #Almacena los eventos que se generaron
    for event in events:                                #Para cada evento
        if event.type==pygame.JOYBUTTONDOWN:            #Si se presionó un botón
            for i in range(buttons):                    #Para cada botón
                if(j.get_button(i)):                    #Si se lee un valor alto
                    print("Button pressed:",i)          #Notificar qué botón se presionó
        if event.type==pygame.JOYHATMOTION:             #Si se presionó el d-pad
            print("D-Pad pressed:",j.get_hat(0))        #Notificar la posición actual del d-pad