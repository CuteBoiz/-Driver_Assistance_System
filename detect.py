import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch

import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

sys.path.append('./yolov5')
from models.experimental import attempt_load
from utils.general import  non_max_suppression

sys.path.append('./EfficientNet_Classifier')
from Efnet.model import EfficientNet

from pose_estimate import PoseEstimator

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def main(face_detect_weight, facial_predict_weight, yawn_classifier_weight, conf_thres=.3, iou_thres=.45):
	EYE_AR_THRESH = 180 # Eye aspect ratio thresh
	EYE_AR_CONSEC_FRAMES = 2 # Nrof consecutive frames eye below the threshold
	EF_COUNTER = 0
	BLINK_COUNTER = 0

	YAWN_THRES = 0.5
	MOUTH_AR_CONSEC_FRAMES = 4
	MF_COUNTER = 0
	YAWN_COUNTER = 0
	
	#Load Models
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	face_detect_model = attempt_load(face_detect_weight, map_location=device)
	facial_predictor = dlib.shape_predictor(facial_predict_weight)

	checkpoint = torch.load(yawn_classifier_weight, map_location='cpu')
	mouth_size = checkpoint['image_size']
	yawn_classifer = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=mouth_size)
	yawn_classifer.load_state_dict(checkpoint['state_dict'])
	yawn_classifer.to(device=device, dtype=torch.float)
	yawn_classifer.set_swish(memory_efficient=False)
	yawn_classifer.eval()
	
	# Check if camera opened successfully
	cap = cv2.VideoCapture(0)
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	pose_estimator = PoseEstimator(img_size=(height, width))
	
	with torch.no_grad():
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == True:
				ori_h, ori_w = frame.shape[:2]
				img_padded = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
				img_padded = img_padded.transpose((2, 0, 1))
				img_padded = np.ascontiguousarray(img_padded)
				img = torch.from_numpy(img_padded).to(device)
				img = img.float()
				img /= 255.0

				if len(img.shape) == 3:
					img = img[None]
				results = face_detect_model(img)
				preds = non_max_suppression(results[0], conf_thres, iou_thres, max_det=1)
				for pred in preds:
					if not len(pred):
						break
					for box in pred:
						x1, y1 = int(box[0]/640*ori_w), int(box[1]/640*ori_h)
						x2, y2 = int(box[2]/640*ori_w), int(box[3]/640*ori_h)
						gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						# Get facial landmark
						shape = facial_predictor(gray, dlib.rectangle(x1, y1, x2, y2))
						shape = face_utils.shape_to_np(shape)

						# Extract boxes
						nose_pts = shape[30:36]
						x1_n, y1_n = np.amin(nose_pts, axis=0)
						x2_n, y2_n = np.amax(nose_pts, axis=0)
						left_eye_pts = shape[42:48]
						x1_le, y1_le = np.amin(left_eye_pts, axis=0)
						x2_le, y2_le = np.amax(left_eye_pts, axis=0)
						right_eye_pts = shape[36:42]
						x1_re, y1_re = np.amin(right_eye_pts, axis=0)
						x2_re, y2_re = np.amax(right_eye_pts, axis=0)
						mouth_pts = shape[48:68]
						x1_m, y1_m = np.amin(mouth_pts, axis=0)
						x2_m, y2_m = np.amax(mouth_pts, axis=0)

						# Blink Counter
						leftEAR = eye_aspect_ratio(left_eye_pts)
						rightEAR = eye_aspect_ratio(right_eye_pts)
						both_ear = ((leftEAR + rightEAR) / 2)* 1000
						if both_ear < EYE_AR_THRESH:
							EF_COUNTER += 1
						else:
							if EF_COUNTER > EYE_AR_CONSEC_FRAMES:
								BLINK_COUNTER += 1
							EF_COUNTER = 0
						
						# Yawn counter
						mouth = frame[y1_m:y2_m, x1_m:x2_m, ]
						mouth = cv2.resize(mouth, (mouth_size, mouth_size), interpolation = cv2.INTER_AREA)
						mouth = np.float32(mouth)
						mouth = mouth*(1/255)
						mean = [0.485, 0.456, 0.406]
						std = [0.229, 0.224, 0.225]
						mouth = (mouth - mean) / std
						mouth = mouth.transpose((2, 0, 1))
						mouth = np.asarray([mouth]).astype(np.float32)
						mouth = torch.from_numpy(mouth).to(device=device, dtype=torch.float)
						yawn_predict = np.squeeze(yawn_classifer(mouth).cpu().softmax(1).numpy())[1]
						if yawn_predict > YAWN_THRES:
							MF_COUNTER += 1
						else:
							if MF_COUNTER > MOUTH_AR_CONSEC_FRAMES:
								YAWN_COUNTER += 1
							MF_COUNTER = 0
						#Get Pose
						pose = pose_estimator.solve_pose_by_68_points(shape)
						pose_estimator.draw_axes(frame, pose[0], pose[1])

						cv2.putText(frame, f"EAR : {both_ear:.2f}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
						cv2.putText(frame, f"BLINK TIME : {BLINK_COUNTER}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
						cv2.putText(frame, f"YAWNING TIME : {YAWN_COUNTER}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

						for (sX, sY) in shape:
							cv2.circle(frame, (sX, sY), 1, (255, 0, 255), -1)
						frame = cv2.rectangle(frame, (x1_n, y1_n), (x2_n, y2_n), (255, 0, 255), 1) # Nose
						frame = cv2.rectangle(frame, (x1_le, y1_le), (x2_le, y2_le), (255, 0, 255), 1) # Left eye
						frame = cv2.rectangle(frame, (x1_re, y1_re), (x2_re, y2_re), (255, 0, 255), 1) # Right eye
						frame = cv2.rectangle(frame, (x1_m, y1_m), (x2_m, y2_m), (255, 0, 255), 1) # Mouth
						frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
				cv2.imshow('Frame',frame)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else: 
				break

	cap.release()
	cv2.destroyAllWindows()

		

if __name__ == '__main__':
	main(face_detect_weight='../weights/face.pt', 
		facial_predict_weight='../weights/shape_predictor_68_face_landmarks.dat', 
		yawn_classifier_weight='../weights/yawning.pth')