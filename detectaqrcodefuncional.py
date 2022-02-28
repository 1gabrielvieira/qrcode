from pyzbar import pyzbar
import cv2
import numpy as np

# Read image
imgbruta = cv2.imread("imgs/photo2.png")
img1 = imgbruta

# Find X,Y coordinates of white pixels
wY, wX = np.where(np.all(imgbruta>=[200,200,200],axis=2))

# Find the corner pixels
top, bottom = wY[0], wY[-1]
left, right = wX[0], wX[-1]
print(top,bottom,left,right)

# Cut the region of qrcode
cut_qrcode = img1[top:(bottom), (left-13):(right+1)]

img2 = cut_qrcode

# resize image
 
scale_percent = 400 
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
  
resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

img=resized

# Apply Super resolution

sr = cv2.dnn_superres.DnnSuperResImpl_create()

path = "FSRCNN_x3.pb"

sr.readModel(path)

sr.setModel("fsrcnn",3)

result = sr.upsample(img)

image = result

# Locate the barcode in the image and decode it
barcodes = pyzbar.decode(image)

# Cycle detect barcodes
for barcode in barcodes:

 # Draw the bounding box of the barcode in the image
 (x, y, w, h) = barcode.rect
 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

 # The barcode data is a byte object, so if we want to print it on the output image
 # To draw it, you need to convert it into a string first
 barcodeData = barcode.data.decode("utf-8")
 barcodeType = barcode.type

 # Draw the barcode data and barcode type on the image
 text = "{} ({})".format(barcodeData, barcodeType)
 cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
  0.5, (0, 0, 255), 2)

 # Print barcode data and barcode type to the terminal
 print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

# Show output image
cv2.imshow("Image", image)
cv2.waitKey(0)
