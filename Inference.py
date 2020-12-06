from MyModels.models import mobilenet_v2_custom,load_image,inferenceImage,inference_np
import cv2
from keras.applications.mobilenet import preprocess_input
from config import CLASSES
import numpy as np
import time

def inference_video(model,video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    i = 0
    ret, frame = cap.read()
    r = cv2.selectROI('a', frame)
    print(r)
    roi_x1 = int(r[0])
    roi_x2 = int(r[0] + r[2])
    roi_y1 = int(r[1])
    roi_y2 = int(r[1] + r[3])
    prev_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    cv2.destroyWindow('a')
    msg = " "
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            # print("time stamp current frame:", i / fps)
            crop = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            diff = cv2.absdiff(prev_crop, crop)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
            i=i+1

            white_pixels = (len(thresh[thresh >= 50]))
            black_pixels = (len(thresh[thresh <= 50]))
            noise = round((white_pixels / thresh.size * 100), 2)

            font = cv2.FONT_HERSHEY_DUPLEX

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if (noise >= 20):
                p=inference_np(model,crop.copy())
                predicted_class = CLASSES[np.argmax(p)]
                confidence_score = np.round(np.amax(p) * 100, 2)
                print(predicted_class,confidence_score)
                if(confidence_score >=90):
                    if predicted_class!="66338-SAUSAGE ROLLS":
                        msg =  str(predicted_class) #", Score : " + str(confidence_score)
                else:
                    msg=" "
            frame = cv2.putText(frame, str(msg), (25, 25), font, 0.75, (255, 255, 0), 1)

            if (i % int(fps*2) == 0):
                msg = " "
                prev_crop = crop.copy()
            cv2.imshow('Frame', frame)
            # cv2.imshow('Thresh', thresh)
            cv2.imshow('crop', crop)

        except Exception as e:
                print(str(e))
                break

if __name__ == '__main__':
    WEIGHTS_TO_LOAD = 'Weights\mobilenet2_weights\weights_12-0.23.hdf5'
    model = mobilenet_v2_custom(len(CLASSES))
    model.load_weights(WEIGHTS_TO_LOAD)
    inference_video(model,'Test Videos/misc.mp4')


