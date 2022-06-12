import cv2
import timeit

# 영상 검출기
def videoDetector(cam):
    global face, eye, leftear, rightear, mouth, nose
    
    
    while True:
        
        start_t = timeit.default_timer() # 알고리즘 시작 시점
        """ 알고리즘 연산 """
        ret,img = cam.read()  # 캡처 이미지 불러오기
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)        # 영상 압축
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 그레이 스케일 변환

        # cascade 얼굴 탐지 알고리즘 
        face_result = face.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.3,# 이미지 피라미드 스케일 factor
                                           minNeighbors=3,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )
        
        eye_result = eye.detectMultiScale(gray,
                                             scaleFactor= 1.3,
                                             minNeighbors=6,
                                             minSize=(20,20)
                                             )
        
        leftear_result = leftear.detectMultiScale(gray,
                                             scaleFactor= 1.1,
                                             minNeighbors=6,
                                             minSize=(20,20)
                                             )
        
        rightear_result = rightear.detectMultiScale(gray,
                                             scaleFactor= 1.1,
                                             minNeighbors=6,
                                             minSize=(20,20)
                                             )
        
        mouth_result = mouth.detectMultiScale(gray,
                                             scaleFactor= 1.3,
                                             minNeighbors=6,
                                             minSize=(20,20)
                                             )
        
        nose_result = nose.detectMultiScale(gray,
                                             scaleFactor= 1.3,
                                             minNeighbors=6,
                                             minSize=(20,20)
                                             )
                                                                           
        for box in face_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), thickness=2)
            cv2.putText(img, "Detected Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255,0,0))
            
        for box in eye_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            cv2.putText(img, "Detected Eye", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0))
            
        for box in leftear_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness=2)
            cv2.putText(img, "Detected Left Ear", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
            
        for box in rightear_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness=2)
            cv2.putText(img, "Detected Right Ear", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255))
            
        for box in mouth_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), thickness=2)
            cv2.putText(img, "Detected Mouth", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0))
                    
        for box in nose_result:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), thickness=2)
            cv2.putText(img, "Detected Nose", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255))
     
        """ 알고리즘 연산 """ 
        terminate_t = timeit.default_timer()         # 알고리즘 종료 시점
        FPS = 'fps' + str(int(1./(terminate_t - start_t )))
        cv2.putText(img,FPS,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        
        
         # 영상 출력
        
        cv2.imshow('facenet',img)
        
        if cv2.waitKey(1) > 0: 
  
            break

def onChange(pos):
    global capture
    err, img = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
    cv2.imshow("facenet", img)
    pass

# 가중치 파일 경로
face_cascade = 'haarcascade_frontalface_alt.xml'
eye_cascade = 'haarcascade_eye.xml'
leftear_cascade = 'haarcascade_mcs_leftear.xml'
rightear_cascade = 'haarcascade_mcs_rightear.xml'
mouth_cascade = 'haarcascade_mcs_mouth.xml'
nose_cascade = 'haarcascade_mcs_nose.xml'

# 모델 불러오기
face = cv2.CascadeClassifier(face_cascade)
eye = cv2.CascadeClassifier(eye_cascade)
leftear = cv2.CascadeClassifier(leftear_cascade)
rightear = cv2.CascadeClassifier(rightear_cascade)
mouth = cv2.CascadeClassifier(mouth_cascade)
nose = cv2.CascadeClassifier(nose_cascade)

# 영상 파일 
capture = cv2.VideoCapture('sample2.mp4')
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.namedWindow('facenet')


# 영상 탐지기
cv2.createTrackbar('length', 'facenet', 0, length, onChange)

onChange(0)
cv2.waitKey()

cap_len = cv2.getTrackbarPos('length', 'facenet')

capture.set(cv2.CAP_PROP_POS_FRAMES,cap_len)

videoDetector(capture)    
    