import cv2
import glob
import numpy as np
from numpy.lib.function_base import append
from pyzbar.pyzbar import decode

# Função utilizada para carregar e passar cada imagem da pasta por vez
path = glob.glob("imgs\*.png")
for image in path:

    # Lê a imagem e cria uma cópia  
    img = cv2.imread(image)
    copia_img = img.copy()

    # Aplica filtros para identificar o contorno das plataformas na imagem
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop para identificar os contornos
    for c in contours:
     perimeter = cv2.arcLength(c,True)

     if perimeter > 700 and perimeter < 1000:  # Condição criada para pegar apenas as plataformas
         #cv2.drawContours(copia_img, [c], 0, (0,0,255), 3) #Caso queira desenhar os contornos da plataforma

         # Recortando cada plataforma presente na imagem
         x,y,w,h = cv2.boundingRect(c)
         crop1 = copia_img[y:y+h, x:x+w]
         crop1_copia = crop1.copy()

         # Aplicando filtros e binarizando a imagem para o primeiro estágio de identificação dos contornos do QRCode
         imgray1 = cv2.cvtColor(crop1_copia,cv2.COLOR_BGR2GRAY)
         thresh1 = cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

         contours2, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

         # Loop para identificar os contornos do QRCode
         for c2 in contours2:
             perimeter2 = cv2.arcLength (c2, True)

             if perimeter2 > 100:  # Condição criada para otimizar a identificação do contorno
                 #cv2.drawContours(crop1_copia, [c], 0, (0,0,255), 3) #Caso queira desenhar os contornos

                 approx =  cv2.approxPolyDP(c2, 0.002*perimeter2, True) # Aproximação de polígono para otimizar o corte da imagem

                 # Recortando o primeiro estágio de recorte do QRCode
                 x,y,w,h = cv2.boundingRect(c2)
                 crop2 = crop1_copia[y:y+h, x:x+w] 
                 crop2 = cv2.resize(crop2, (crop2.shape[1]*4,crop2.shape[0]*4), interpolation=cv2.INTER_CUBIC) # dando resize para facilitar a visualizaçao do qrcode
                 
                 # Aplicando filtros mais complexos e binarizando a imagem para o segundo estágio de recorte do contorno do QRCode
                 B1 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
                 BL1 = np.array([0,0,70])
                 BH1 = np.array([0,0,255])
                 mascara = cv2.inRange(B1, BL1, BH1)
                 _, threshb = cv2.threshold(mascara, 0,255, cv2.THRESH_BINARY)
                 edges = cv2.Canny(threshb, 20, 250)

                 contours3, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                 
                 boxes = []

                 # Condição criada para evitar erros de zero-size na imagem
                 if len(contours3) > 3:
                    
                    # Loop criado para o último estágio de recorte do QRCode, agora será feito o recorte retangular ao redor de todos os contornos apresentados
                    for c3 in contours3:
                        x,y,w,h = cv2.boundingRect(c3)
                        boxes.append([x,y, x+w, y+h])

                    boxes = np.asarray(boxes)
                    left, top = np.min(boxes, axis=0)[:2]
                    right, bottom = np.max(boxes, axis=0)[2:]
                    #cv2.rectangle(crop2, (left, top), (right, bottom), (0,0,255), 2) # Caso queira desenhar a região
                    crop3 = crop2[top:bottom, left:right]

                    # Agora, por fim, foi realizado a decodificação do QRCode
                    qrcodes = decode(crop3)

                    for qrcode in qrcodes:
                         print('Informação do QRCode:', qrcode.data.decode("utf-8")) 

                         cv2.imshow("fotos", crop2) 
                         cv2.waitKey(0)
                         cv2.destroyAllWindows()