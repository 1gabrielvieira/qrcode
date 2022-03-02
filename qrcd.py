import cv2
import glob
import numpy as np
from numpy.lib.function_base import append
from pyzbar.pyzbar import decode

img = cv2.imread('imgs/photo1.png')
copia_img = img.copy()

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(imgray, 127, 255, 0)

thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

 perimeter = cv2.arcLength(c,True)

 if perimeter > 700 and perimeter < 1000:
     #cv2.drawContours(copia_img, [c], 0, (0,0,255), 3) #Caso queira desenhar os contornos
     x,y,w,h = cv2.boundingRect(c)
     cropped = img[y:y+h, x:x+w]

     #Aplicando filtros para retirar os rúidos e manter apenas o qrcode
     black4 = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
     black_lo4 = np.array([0,0,0])
     black_hi4 = np.array([0,0,255])
     mask4 = cv2.inRange(black4, black_lo4, black_hi4)
     # realizando uma binarizaçao da imagem para que seja possivel traçar as bordas e assim conseguir os contornos das partes brancas do qrcode
     _, thresh2 = cv2.threshold(mask4, 0,255, cv2.THRESH_BINARY)

     contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     for c2 in contours2:
        perimeter2 = cv2.arcLength(c2, True)

        if perimeter2 > 100:
            approx =  cv2.approxPolyDP(c2, 0.002*perimeter2, True)
            #cv2.drawContours(cropped, [c2],0 , (0,255,0), 3) # Caso queira desenhar os contornos
            x,y,w,h = cv2.boundingRect(approx)
            cropped2 = cropped[y:y+h, x:x+w]

            scale_percent = 500 
            width = int(cropped2.shape[1] * scale_percent / 100)
            height = int(cropped2.shape[0] * scale_percent / 100)
            dim = (width, height)
  
            resized = cv2.resize(cropped2, dim, interpolation = cv2.INTER_AREA)
            
            # Caso queria super resolução
            #sr = cv2.dnn_superres.DnnSuperResImpl_create()
            #path = "FSRCNN_x3.pb"
            #sr.readModel(path)
            #sr.setModel("fsrcnn",3)
            #result = sr.upsample(resized)


            black5 = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            black_lo5 = np.array([0,0,70])
            black_hi5 = np.array([0,0,255])
            mask5 = cv2.inRange(black5, black_lo5, black_hi5)
            # realizando uma binarizaçao da imagem para que seja possivel traçar as bordas e assim conseguir os contornos das partes brancas do qrcode
            _, thresh3 = cv2.threshold(mask5, 0,255, cv2.THRESH_BINARY)

            edged = cv2.Canny(thresh3, 20, 250)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            opening = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel, iterations=4)

            contours3, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []

            if len(contours3) > 3: # condiçao, pois algumas imagens existiam bordas sem qrcode na imagem
                for c3 in contours3:
                    (x, y, w, h) = cv2.boundingRect(c3)
                    boxes.append([x,y, x+w,y+h])

                boxes = np.asarray(boxes)
                left, top = np.min(boxes, axis=0)[:2]
                right, bottom = np.max(boxes, axis=0)[2:]
                cropped3 = resized[top:bottom, left:right]

                # Caso queria super resolução
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                path = "FSRCNN_x3.pb"
                sr.readModel(path)
                sr.setModel("fsrcnn",3)
                result = sr.upsample(cropped3)

                print(decode(result))
 
                cv2.imshow('Imagem', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()