
# Programa de carga de un mapa y su transformación en un grid-map #

import numpy as np 
import cv2
import sys, getopt
import os.path
from numpy import savetxt

def nothing(x):
   pass 

# Argumentos de entrada 

argv = sys.argv[1:]


try:
    opts, args = getopt.getopt(argv,"dc")
    
except:
    print("\nPrograma de carga y transformacion de un mapa en una máscara binaria")
    print("\Formato: python map-adquisition.py [MODO] [PATH_TO_IMAGE]\n")
    print("\nModos:\n")
    print("\t-d : modo default con los parámetros del Ypacarai normales.")
    print("\t-c : modo calibracion para otros casos.\n")
    exit()

default_mode = 0

for opt,args in opts:
    if opt == '-d':
        print("Seleccionando modo default.")
        default_mode = 1
    elif opt in "-c":
        print("Seleccionando modo calibracion.")
        default_mode = 0
    else:
        exit()
    



# Cargamos la imagen #
img = cv2.imread('YpacarayMap_color.png')


scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height) 
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

img = cv2.resize(img,(240,240))

# Pasamos al espacio HSV para sacar los colores #
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

if(default_mode == 0):
    # Creamos los sliders para la calibraion #
    cv2.namedWindow('Parametros')
    cv2.createTrackbar('Hue Minimo','Parametros',0,179,nothing)
    cv2.createTrackbar('Hue Maximo','Parametros',0,179,nothing)
    cv2.createTrackbar('Saturation Minimo','Parametros',0,255,nothing)
    cv2.createTrackbar('Saturation Maximo','Parametros',0,255,nothing)
    cv2.createTrackbar('Value Minimo','Parametros',0,255,nothing)
    cv2.createTrackbar('Value Maximo','Parametros',0,255,nothing)

while(True):

    if(default_mode == 0):
        #Leemos los sliders y guardamos los valores de H,S,V para construir los rangos:
        hMin = cv2.getTrackbarPos('Hue Minimo','Parametros')
        hMax = cv2.getTrackbarPos('Hue Maximo','Parametros')
        sMin = cv2.getTrackbarPos('Saturation Minimo','Parametros')
        sMax = cv2.getTrackbarPos('Saturation Maximo','Parametros')
        vMin = cv2.getTrackbarPos('Value Minimo','Parametros')
        vMax = cv2.getTrackbarPos('Value Maximo','Parametros')

        #Creamos los arrays que definen el rango de colores:
        lower_tresh = np.array([hMin,sMin,vMin])
        upper_tresh=np.array([hMax,sMax,vMax])
    
    else:
        lower_tresh = np.array([82,109,150])
        upper_tresh = np.array([125,154,222])

    # Máscara resultante #
    mask = cv2.inRange(hsv_img, lower_tresh, upper_tresh)
    mask_filtered = cv2.medianBlur(mask,int(scale_percent/100*15))

    # Cropeamos la máscara sobre el mapa original
    crop = cv2.bitwise_and(img, img, mask=mask_filtered)

    #Mostramos los resultados y salimos:
    cv2.imshow('Original',img)
    cv2.imshow('Mascara',mask_filtered)
    cv2.imshow('Superposicion',crop)


    # Esperamos a salir#
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.imwrite('MASK.png',mask_filtered)

# Guardamos el resultado


# Realizamos el resample de la máscara #

ancho_mapa_m = 15000 # metros #
alto_mapa_m = 13000 # metros #

resolucion_m_each_cell = 500 # Cada celda ocupa resolucion_m_each_cell metros de lado #

width_scale = round(ancho_mapa_m/resolucion_m_each_cell)
height_scale = round(alto_mapa_m/resolucion_m_each_cell)

new_size = (width_scale,height_scale)
grid_map =  cv2.resize(mask_filtered,new_size,interpolation=cv2.INTER_NEAREST)

# Con el mapa resampleado, hacemos hacemos un crop para reducir el numero de casillas no posibls #

x_f,y_f,w_f,h_f = cv2.boundingRect(grid_map)

print("Se va a reescalar el mapa con {} celdas de alto y {} celdas de ancho.".format(width_scale,height_scale))

celdas_AGUA = cv2.countNonZero(grid_map)
celdas_total = width_scale*height_scale
celdas_TIERRA = celdas_total-celdas_AGUA

print("En total, tendremos {} celdas de AGUA (BLANCAS) y {} celdas de TIERRA (NEGRAS). {} celdas en total.".format(celdas_AGUA,celdas_TIERRA,celdas_total))

cropped_image = np.zeros((h_f+1,w_f+1))
cropped_image = grid_map[y_f:y_f+h_f,x_f:x_f+w_f]/255
cropped_image = cv2.copyMakeBorder(cropped_image,1,1,1,1, borderType = cv2.BORDER_CONSTANT, value = 0)

cv2.namedWindow('Mapa de celdas',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mapa de celdas', 600,600)
cv2.imshow('Mapa de celdas',cropped_image )

while(True):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

if(default_mode == 1):
	savetxt('YpacaraiMap.csv', cropped_image , delimiter=',', fmt = '%u')
	print("Archivo CSV creado!")
else:
	savetxt('YpacaraiMap_calibrated.csv', cropped_image , delimiter=',', fmt = '%u')
	print("Archivo CSV creado!")
