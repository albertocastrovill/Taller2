from gpiozero import Motor                  #importar módulo de motores de libreria gpiozero
import curses                               #importar librería curses

motor=Motor(17,18)                          #Definimos que nuestro motor está conectado a GPIO17 y GPIO18

actions={                                   #Lista de acciones a realizar dependiendo de la tecla que se presione
    curses.KEY_UP:      motor.forward,      #Flecha hacia arriba mueve el motor hacia "adelante"
    curses.KEY_DOWN:    motor.backward,     #Flecha habia abajo lo mueve hacia "atrás"
    119:                motor.forward,      #Letra "w" (código ascii decimal 119) funciona igual que flecha hacia arriba
    115:                motor.backward      #Letra "s" (código ascii decimal 115) funciona igual que flecha hacia arriba
}

def main(window):                           #Parte principal, se abre una ventana que de momento no nos servirá
    next_key=None                           #Inicializamos definiendo que no se ha presionado ninguna tecla
    while True:                             #Loop eterno
        curses.halfdelay(1)                 #Damos una décima de segundo para no saturar al sistema
        if next_key is None:                #Si la última vez que revisamos no se había presionado ninguna tecla...
            key=window.getch()              #Almacenamos el valor de la tecla que se presionó (puede ser None)
        else:                               #Si la última vez que revisamos sí se había presionado una tecla
            key=next_key                    #Definimos la tecla actual como la
            next_key=None

        if key!=-1:                         #Si sí se presionó una tecla (ninguna presión se almacena como -1)
            curses.halfdelay(3)             #Esperamos tres décimas de segundo para no saturar al sistema
            action=actions.get(key)         #Definimos la acción a definir a partir de la lista de acciones inicial y la tecla presionada
            if action is not None:          #Si sí se encontró una acción (se utilizaron las teclas correctas)
                action()                    #Realizamos la acción
            else:                           #Si se presionó una tecla que no tenemos definida
                motor.stop()                #Detenemos el motor
            next_key=key                    #Definimos la tecla actual como la siguiente, porque ya se realizó su acción
            while next_key==key:            #Mientras ambas variables sean iguales
                next_key=window.getch()     #Leemos el teclado esperando que cambie la entrada
        else:                               #Si no hubo presión de teclado
            motor.stop()                    #Detenemos el motor

curses.wrapper(main)                        #Función que internamente corre nuestra main, reestablece teclador/pantalla cuando hay error