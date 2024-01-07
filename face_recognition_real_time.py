import cv2
import time
from ultralight import UltraLightDetector
from faiss_detect import predict_face
pTime = 0 
cTime = 0
video_capture ='./imgvideo2.mp4'
# cap = cv2.VideoCapture(0) #camera capture
cap = cv2.VideoCapture(video_capture) #video capture
detector = UltraLightDetector(providers=['CPUExecutionProvider'])
while True:
    success, img = cap.read()
    boxes, scores = detector.detect_one(img)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    i = 0
    for  box in boxes:
        x, y, w, h = box
        cv2.rectangle(img,(x,y),(w,h),(255,0,255),4)
        imgCrop = img[y:h,x:w]
        try:
            img_resized = cv2.resize(imgCrop,(160, 160))
        except:
            continue
        a = predict_face(img_resized)
        cv2.putText(img,a,(int(box[0]), int(box[1]) - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,  (0, 0, 255),2)

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # Ấn Esc để thoát
            break

cap.release()
cv2.destroyAllWindows()