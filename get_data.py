import cv2
import time
from ultralight import UltraLightDetector
from faiss_detect import predict_face
pTime = 0 
cTime = 0
cap = cv2.VideoCapture(0)
detector = UltraLightDetector(providers=['AzureExecutionProvider'])
counter = 0
folder = 'data_face/tphu/'
while True:
    success, img = cap.read()
    boxes, scores = detector.detect_one(img)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    try:
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img,(x,y),(w,h),(255,0,255),4)

            imgCrop = img[y:h,x:w]
            img_resized = cv2.resize(imgCrop,(160, 160))

            cv2.imshow('Crop',img_resized)
    except:
        continue

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter  += 1
        cv2.imwrite(f'{folder}/image_{counter}_tphu.jpg',img_resized)
        print(counter)
    if key & 0xFF == 27:  # Ấn Esc để thoát
            break

cap.release()
cv2.destroyAllWindows()