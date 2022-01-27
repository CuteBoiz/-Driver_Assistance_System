import argparse
import os
import sys
import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist

import face_alignment

sys.path.append('./EfficientNet_Classifier')
from Efnet.model import EfficientNet
from utils.report_utilities import infer_preprocess

from pose_estimate import PoseEstimator


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	if not C:
		return 1000
	ear = (A + B) / (2.0 * C)
	return ear


def main(args):
	assert os.path.isfile(args.yawn_weight)
	assert os.path.isfile(args.pose_file)
	assert args.eye_thres > 0
	assert args.mouth_thres > 0
	assert args.eye_close_consec > 1
	assert args.mouth_open_consec > 1

	# Initialzie thres
	EYE_AR_THRESH = args.eye_thres
	EYE_AR_CONSEC_FRAMES = args.eye_close_consec
	EF_COUNTER = 0
	BLINK_COUNTER = 0

	YAWN_THRES = args.mouth_thres
	MOUTH_AR_CONSEC_FRAMES = args.mouth_open_consec
	MF_COUNTER = 0
	YAWN_COUNTER = 0
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Load models
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

	checkpoint = torch.load(args.yawn_weight, map_location='cpu')
	mouth_size = checkpoint['image_size']
	yawn_classifer = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=mouth_size)
	yawn_classifer.load_state_dict(checkpoint['state_dict'])
	yawn_classifer.to(device=device, dtype=torch.float)
	yawn_classifer.set_swish(memory_efficient=False)
	yawn_classifer.eval()

	# Read Camera
	cap = cv2.VideoCapture(0)
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	pose_estimator = PoseEstimator(img_size=(height, width), model_path=args.pose_file)

	# Check if camera opened successfully
	with torch.no_grad():
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == True:
				preds = fa.get_landmarks(frame)
				if preds is not None:
					for pred in preds:
						if pred is None:
							break
						pred = np.array(pred, dtype=int)
						
						# Extract features
						left_eye = pred[42:48]
						right_eye = pred[36:42]
						mouth_pts = pred[48:68]
						x1_m, y1_m = np.amin(mouth_pts, axis=0)
						x2_m, y2_m = np.amax(mouth_pts, axis=0)

						# Detect eye blinking
						leftEAR = eye_aspect_ratio(left_eye)
						rightEAR = eye_aspect_ratio(right_eye)
						both_ear = ((leftEAR + rightEAR) / 2)
						if both_ear < EYE_AR_THRESH:
							EF_COUNTER += 1
						else:
							if EF_COUNTER > EYE_AR_CONSEC_FRAMES:
								BLINK_COUNTER += 1
							EF_COUNTER = 0
						
						# Mouth open classifier
						if y2_m - y1_m > 0 and x2_m - x1_m > 0:
							mouth = frame[y1_m:y2_m, x1_m:x2_m, ]
							mouth = infer_preprocess(mouth, mouth_size)
							mouth = torch.from_numpy(mouth).to(device=device, dtype=torch.float)
							yawn_predict = np.squeeze(yawn_classifer(mouth).cpu().softmax(1).numpy())[1]
							if yawn_predict > YAWN_THRES:
								MF_COUNTER += 1
							else:
								if MF_COUNTER > MOUTH_AR_CONSEC_FRAMES:
									YAWN_COUNTER += 1
								MF_COUNTER = 0

						# Head Pose estimation
						pose = pose_estimator.solve_pose_by_68_points(pred)
						pose_estimator.draw_axes(frame, pose[0], pose[1])
					
					# Visualize
					for point in pred:
						frame = cv2.circle(frame, (point[0], point[1]), 1, (255, 0, 255), -1)
					frame = cv2.flip(frame, 1)
				cv2.putText(frame, f"BLINK TIME : {BLINK_COUNTER}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
				cv2.putText(frame, f"YAWN TIME : {YAWN_COUNTER}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
				cv2.imshow('Frame',frame)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-et', '--eye_thres', type=float, default=0.22, help='eye aspect ratio threshold')
	parser.add_argument('-ecc', '--eye_close_consec', type=int, default=2, help='number of eye close frame consecutive')
	parser.add_argument('-mt', '--mouth_thres', type=float, default=0.5, help='yawn classifier threshold')
	parser.add_argument('-moc', '--mouth_open_consec', type=int, default=6, help='number of mouth open frame consecutive')

	parser.add_argument('-yw', '--yawn_weight', type=str, default='../weights/yawning.pth', help='path to yawning classifier model')
	parser.add_argument('-pf', '--pose_file', type=str, default='../weights/head_pose.txt' , help='path to head_pse.txt file')
	args = parser.parse_args()
	main(args)