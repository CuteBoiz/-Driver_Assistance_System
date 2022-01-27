'''

author: phatnt
date modified: 2021-09-06
'''

import numpy as np
import os
import torch
import xlsxwriter
from torch import nn
import cv2
from tqdm.autonotebook import tqdm

def softmax(x):
	'''
		Calculate softmax of an array.
		Args:
			x: an array.
		Return:
			Softmax score of input array.
	'''
	assert x is not None, '[ERROR]: input is none!'
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)


def infer_preprocess(data, image_size):
	if isinstance(data, np.ndarray):
		data = [data]
	elif isinstance(data, list):
		pass		
	for i in range(len(data)):
		image = data[i]
		image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
		image = np.float32(image)
		image = image*(1/255)
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		image = (image - mean) / std
		data[i] = image.transpose((2, 0, 1))
	data = np.asarray(data).astype(np.float32)
	return data


def get_cofusion_matrix(evaluate_dict):
	'''
		Get confusion matrix from test set.
		Args:
			evaluate_dict: dictionary for evaluate include:
				+ model: testing model.
				+ train_classes: trained classes.
				+ device: testing device.
				+ threshold: class threshold array.
				+ data_loader: torch data_loader format.
		Return:
			accuracy: total accuracy base on test set.
			result_matrix: confusion matrix
			false_defects: a list of false_defects to generate image in report file/
			underkill: number of underkill images.
			overkill:: number of overkill images.
	'''
	assert evaluate_dict is not None, '[ERROR]: evaluate dict is none!'
	model = evaluate_dict['model']
	train_classes = evaluate_dict['train_classes']
	device = evaluate_dict['device']
	thresholds = evaluate_dict['thresholds']
	data_loader = evaluate_dict['data_loader']
	nrof_classes = len(train_classes)
	result_matrix = np.zeros((nrof_classes, nrof_classes + 1), dtype=np.int32) 
	overkill = 0
	underkill = 0
	false_defects = []
	new_thresholds = np.ones(len(train_classes))
	process_bar = tqdm()

	#Caculate confusion matrix
	model.eval()
	with torch.no_grad():
		for i, sample_batched in enumerate(data_loader):
			#Get output from model
			images = sample_batched['image']
			path = sample_batched['path']
			target = sample_batched['target']
			images = images.to(device=device, dtype=torch.float)
			target = np.array(target)
			score = model(images)
			score = score.cpu().data.numpy()

			#Compare with target
			for i in range (len(target)):
				gt_label = target[i]
				softmaxed = softmax(score[i])
				highest_score_idx = softmaxed.argmax()
				#Apply theshold mask
				mask = (softmaxed >= thresholds)
				if not any(mask):
					predicted_label = nrof_classes
				else:
					masked = softmaxed * mask
					predicted_label = masked.argmax()
				#Cofusion matrix
				result_matrix[gt_label][predicted_label] += 1
				
				if gt_label != predicted_label: #False predict
					false_defect = {'image_path': path[i],
									'gt_label': gt_label,
									'predicted_label': predicted_label,
									'gt_score': softmaxed[gt_label],
									'highest_label': highest_score_idx,
									'highest_score': softmaxed[highest_score_idx]}
					false_defects.append(false_defect)
					if predicted_label < nrof_classes:
						if train_classes[gt_label] == 'Pass':
							overkill += 1
						if train_classes[predicted_label] == 'Pass':
							underkill += 1
					process_bar.set_description("False Predicts: {}".format(len(false_defects)))
				elif gt_label == predicted_label: #True predict
					if (softmaxed[gt_label] < new_thresholds[gt_label]):
						new_thresholds[gt_label] = softmaxed[gt_label]
				
	count = 0
	for i in range(nrof_classes):
		count+=result_matrix[i][i]
	accuracy = (count/np.sum(result_matrix))*100
	process_bar.close()
	print('New thresholds:', new_thresholds)
	return accuracy, result_matrix, false_defects, underkill, overkill

def calculate_exel_char(start_char, length):
	'''
		Caculate column character(s) of excel in order to expand the width of those column.
		Args:
			start_char: starting column.
			length: number of columns that need to be expand width.
		Return:
			result: excel's column character(s).
	'''
	start_order = ord(start_char)
	end_order = start_order + length
	over = False
	first_char_ord = ord('@')
	while (end_order > ord('Z')):
		over = True
		first_char_ord += 1
		end_order -= 26
	if over:
		result ='{}{}'.format(chr(first_char_ord), chr(end_order))
	else:
		result = chr(end_order)
	return result

def set_cols(sheet, list_col_width):
	'''
		Set column(s) width from calculated char.
		Args:
			sheet: excel sheet.
			list_col_width: list column(s) and their width.
	'''
	for col,width in list_col_width:
		if len(col) == 1:
			sheet.set_column('{}:{}'.format(col, col), width)
		elif len(col) > 1:
			sheet.set_column('{}'.format(col), width)
			

def create_report(data_folder, evaluate_dict, report_save_dir):
	'''
	Create report sheets.
		+ Sheet 1: Confusion matrix.
		+ Sheet 2: Quantities (train/val/test).
		+ Sheet 3: Underkill images.

	'''

	train_classes = evaluate_dict['train_classes']
	thresholds = evaluate_dict['thresholds']

	train_file = os.path.join(data_folder, "train.txt")
	valid_file = os.path.join(data_folder, "valid.txt")
	test_file =  os.path.join(data_folder, "test.txt")

	assert os.path.isfile(train_file)
	assert os.path.isfile(valid_file)
	assert os.path.isfile(test_file)
	assert os.path.isdir(report_save_dir)

	#Get Total quantites from train/test/valid set
	files = [train_file, valid_file, test_file]
	total_quantities = np.zeros((len(files) ,len(train_classes)), dtype=np.int32)
	for i in range(len(files)):
		with open(files[i], 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			label = int(line.split("\"")[-1].strip())
			total_quantities[i][label] += 1

	total_accuracy, result_matrix, false_defects, underkill, overkill = get_cofusion_matrix(evaluate_dict)

	#REPORT
	report_save_path = os.path.join(report_save_dir, 'report.xlsx')
	report_sheet = xlsxwriter.Workbook(report_save_path)

	sheet1 = report_sheet.add_worksheet('TOMOC Report')
	sheet2 = report_sheet.add_worksheet('Defect Quantities')
	sheet3 = report_sheet.add_worksheet('False Predict')

	header_format = report_sheet.add_format({'bold':1, 'border':1, 'align':'center', 'valign':'vcenter', 'fg_color':'#B9CDE5'})
	header_format.set_text_wrap()
	normal_format = report_sheet.add_format({'border':1, 'align':'center', 'valign':'vcenter'})
	normal_format.set_text_wrap()

	#SHEET 1
	sheet1_column_width = [ ('B', 25), # Model's name
							('C', 40), # Defect's name
							('D', 20), # Quantities
							('E:{}'.format(calculate_exel_char('E', len(train_classes)-1)), 30), #Train classes
							('{}:{}'.format(calculate_exel_char('E', len(train_classes)), 
											calculate_exel_char('E', len(train_classes)+2)), 25)] #Thres,Acc
	
	set_cols(sheet1, sheet1_column_width)
	for row in range(1, 4):
		for col in range(1, 7+len(train_classes)):
			sheet1.write(row, col, '', header_format)
	sheet1.freeze_panes(3, 4)
	sheet1.merge_range("E2:{}2".format(calculate_exel_char('E',len(train_classes)-1)), 'Defect Type', header_format)
	sheet1.merge_range("B4:B{}".format(4+len(train_classes)-1), 'Efficient-Net', header_format)

	row_header_1_1 = ['Threshold','Accuracy']
	sheet1.write_row(1, 6+len(train_classes)-1, row_header_1_1, header_format)
	row_header_1_2 = ['Model\'s Name','Defect Type', 'Quantities']
	row_header_1_2.extend(train_classes)
	row_header_1_2.append('Unknown')
	sheet1.write_row(2, 1, row_header_1_2, header_format)


	for i in range (len(train_classes)):
		row_content_1_1 = []
		row_content_1_1.append(train_classes[i])		#Class name
		row_content_1_1.append(total_quantities[2][i])	#Test Quantity of each class
		row_content_1_1.extend(result_matrix[i])		#Predict's result
		row_content_1_1.append(thresholds[i])			#Class threshold
		row_content_1_1.append('{:.2f}%'.format(result_matrix[i][i]/total_quantities[2][i]*100)) #Accuracy
		sheet1.write_row(3+i, 2, row_content_1_1, normal_format)
		sheet1.write(3+i, 4+i, result_matrix[i][i], header_format)
	
	row_content_1_2 = []
	row_content_1_2.append('Underkill')
	row_content_1_2.append(underkill)
	row_content_1_2.append('Overkill')
	row_content_1_2.append(overkill)
	row_content_1_2.append('Total False Predict')
	total_train_img = np.sum(total_quantities[2])
	true_predict = np.sum([result_matrix[i][i] for i in range(len(train_classes))])
	row_content_1_2.append(total_train_img - true_predict)
	row_content_1_2.append('Total Accuracy')
	row_content_1_2.append("{:.2f}%".format(total_accuracy))
	sheet1.write_row(3+len(train_classes) ,len(train_classes)-1, row_content_1_2, header_format)

	#SHEET 2
	sheet2.freeze_panes(3, 2)
	sheet2_column_width = [ ('B', 40), #Defect
							('C:F', 15)] #Other
	set_cols(sheet2, sheet2_column_width)
	sheet2.merge_range("C2:F2", "Quantities", header_format)
	row_header_2_1 = ['Defect Type','Total','Train','Validation','Test']
	sheet2.write_row(2, 1, row_header_2_1, header_format)

	for i in range(len(train_classes)):
		row_content_2 = []
		row_content_2.append(train_classes[i])	#Class name
		class_quantity = [class_[i] for class_ in total_quantities]
		row_content_2.append(np.sum(class_quantity))	#Class quantity
		row_content_2.extend(class_quantity)
		sheet2.write_row(3+i, 1, row_content_2, normal_format)
	row_header_2_2 = ['Total']
	row_header_2_2.append(np.sum(total_quantities))
	row_header_2_2.extend(np.sum(total_quantities, axis=1))
	sheet2.write_row(3+len(train_classes), 1, row_header_2_2, header_format)

	#SHEET 3
	sheet3.set_column('C:I', 34)
	sheet3.freeze_panes(2, 2)
	row_header_3 = ['No', 'False Defects', 'Ground True Label', 'Predicted Label', 'GT Class Score' , 'Highest Predicted Score','Image Path' ]
	sheet3.write_row(1, 1, row_header_3, header_format)
	saved_path = "./temp/"
	os.makedirs(saved_path, exist_ok=True)

	count = 1
	for false_defect in false_defects:
		sheet3.set_row(1+count, 200)
		row_content_3 = []
		row_content_3.append(count) #ID
		row_content_3.append('image')
		row_content_3.append(train_classes[false_defect['gt_label']])	#GT label
		if false_defect['predicted_label'] < len(train_classes):  #Predicted label
			predicted_label = train_classes[false_defect['predicted_label']]
		else:
			predicted_label = 'Unknown'
		row_content_3.append(predicted_label)
		row_content_3.append("{:.3f}".format(false_defect['gt_score'])) #Gt score
		if predicted_label == 'Unknown':
			row_content_3.append('{:.5f}\n({})'.format(false_defect['highest_score'], train_classes[false_defect['highest_label']]))
		else:
			row_content_3.append('{:.5f}'.format(false_defect['highest_score']))
		row_content_3.append(false_defect['image_path'])
		sheet3.write_row(1+count, 1, row_content_3, normal_format)
		#Insert image
		image = cv2.imread(false_defect['image_path'])
		resized = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
		saved_img = os.path.join(saved_path, "{}.bmp".format(count))
		cv2.imwrite(saved_img ,resized)
		sheet3.insert_image(1+count, 2, saved_img ,{'x_scale':1.0 , 'y_scale':1.0, 'x_offset': 5, 'y_offset': 5, 'object_position': 1})
		count+=1

	report_sheet.close()
	print('{} Created'.format(report_save_path))