import cv2
import cvzone
from ultralytics import  YOLO

img = cv2.imread('4.jpg')
img = cv2.resize(img, (500, 600))
model = YOLO('best.pt')
results = model(img)

for result in results:

    cls = result.probs.top1
    conf = result.probs.top1conf
    name = model.names[cls]

    print(f'Diamond Shape : {name}')
    print(f'Confidence Score :  {conf: .2f}')
    cvzone.putTextRect(img, f'{name}, {conf: .2f}', pos=(30, 30), scale=1.5, thickness=3, colorT=(0, 0, 0))

cv2.imshow('Diamond Shape', img)
cv2.waitKey(0)
cv2.destroyAllWindows()