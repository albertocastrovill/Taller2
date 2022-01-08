import pygame
import time

pygame.joystick.init()
pygame.init()

print(range(pygame.joystick.get_count()))

j=pygame.joystick.Joystick(0)
j.init()
print("Joystick inicializado")

name=j.get_name()
print("Name:",name)
axes=4
print("Axes: ",axes)
for i in range (axes):
    axis=j.get_axis(i)
    print ("Axis",i,":",axis)
buttons=j.get_numbuttons()
print("Buttons: ",buttons)
for i in range (buttons):
    button=j.get_button(i)
    print("Button",i,":",button)
hats=j.get_numhats()
print("Hats:",hats)
for i in range (hats):
    hat=j.get_hat(i)
    print("Hat",i,":",hat)
    
oldVal=[0,0,0,0]
while True:

    pygame.event.pump()

    for i in range (axes):
        newVal=j.get_axis(i)
        if newVal!=oldVal[i]:
            print("axis",i,"=",newVal)
            oldVal[i]=newVal
            
            
    event=pygame.event.get()
    for event in event:
        if event.type==pygame.JOYBUTTONDOWN:
            for i in range (buttons):
                if (j.get_button(i)):
                    print("Button pressed:",i)

                    
        if event.type==pygame.JOYHATMOTION:
            print("D-pad pressed:",j.get_hat(0))
            
    #time.sleep(3)
    
# Número de Botónes en PlaySatation COntroller

# 0 = Cuadrado
# 1 = X
# 2 = Circulo
# 3 = Triángulo
# 4 = L1
# 5 = R1
# 6 = L2
# 7 = R2
# 8 = Share
# 9 = Options
# 10 = Joy Izquierdo
# 11 = Joy Derecho

# Axis 0 = Izq / Der - Joy Izquierdo
# Axis 1 = Arr / Abaj - Joy Izquierdo
# Axis 2 = Izq / Der - Joy Izquierdo
# Axis 3 = Arr / Abaj - Joy Izquierdo


