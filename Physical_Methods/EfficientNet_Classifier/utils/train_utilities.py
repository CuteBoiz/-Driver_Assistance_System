'''
Training ultilities.

author: phatnt
date modified: 2022-01-07
'''

import shutil
import os
import torch
import numpy as np
from tqdm.autonotebook import tqdm


def save_checkpoint(state, saved_path, is_best_loss, is_best_acc):
	'''
		Save training model in a specific folder.
		Args:
			state: torch state dict.
			saved_path: model saved place.
			is_best_loss: is this model reach lowest loss?
			is_best_acc: is this model reach highest accuracy on valid set?
	'''
	assert state is not None, '[ERROR]: state dict is none!'
	os.makedirs(saved_path, exist_ok=True)

	saved_file = os.path.join(saved_path, 'model_last.pth') 
	torch.save(state, saved_file)
	if is_best_loss:
		shutil.copyfile(saved_file, saved_file.replace('last', 'best_loss'))
	if is_best_acc:
		shutil.copyfile(saved_file, saved_file.replace('last', 'best_acc'))
	print('[INFO] Checkpoint saved!')


def get_class_weight(train_file, nrof_classes):
	'''
		Get class weight array. In order to deal with imbalance dataset.
		Args: 
			train_file: txt file with format ("image_path", label_id) each line.
			nrof_classes: number of training classes.
		Return:
			An class weight array (length = nrof_classes)
	'''
	assert os.path.isfile(train_file), '[ERROR]: {} not found!'.format(train_file)
	assert isinstance(nrof_classes, int) and nrof_classes >= 0, '[ERROR]: nrof_classes is not valid!'
	
	#Count quantity of each class
	class_quantities = np.zeros(nrof_classes, dtype=np.int32)
	class_weights = []
	with open(train_file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		target = int(line.split("\"")[2].strip())
		assert target < nrof_classes
		class_quantities[target]+=1

	#Caculate class-weight
	for quantity in class_quantities:
		cls_weight = np.sum(class_quantities)/(nrof_classes*quantity)
		class_weights.append(cls_weight)

	return class_weights

def adjust_learning_rate(optimizer, epoch_num, init_lr):
	'''
		Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
		Args:
			optimizer: training optimizer.
			epoch_num: current epoch num.
			learning_rate: init learning rate
	'''
	lr = init_lr * (0.1 ** (epoch_num // 5))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def evaluate(val_loader, model, device):
	'''
		Caculate Accuracy from validation set.
		Args:
			val_loader: torch dataloader format.
			model: Efficient training model.
			device: current training gpu.
		Return:
			accuracy: Accuracy of model base on validation set.
	'''
	print('Evaluating ...')
	model.eval()
	True_Cases = 0
	Cases = 0
	with torch.no_grad():
		for i, sample_batched in enumerate(val_loader):
			images = sample_batched['image']
			target = sample_batched['target']
			images = images.to(device=device, dtype=torch.float)

			output = model(images)
			output = output.cpu().data.numpy().argmax(axis=1)
			target = np.array(target)
			for i in range(len(output)):
				Cases += 1
				if int(output[i]) == int(target[i]):
					True_Cases += 1
		accuracy = (True_Cases/Cases)*100
	return accuracy


def train_eff(model, train_loader, val_loader, optimizer, criterion, scheduler, init_lr, max_epoch, last_epoch, device, writer, arch, image_size, nrof_classes, weight_save_dir):
	best_loss = 1e5
	best_acc = 0
	try:
		for epoch in range(1, max_epoch):
			if epoch < last_epoch:
				continue
			model.train()
			epoch_loss = []
			is_best_loss = False
			is_best_acc = False
			adjust_learning_rate(optimizer, epoch, init_lr)
			progress_bar = tqdm(total=len(train_loader))

			for i, sample_batched in enumerate(train_loader):
				images = sample_batched['image']
				target = sample_batched['target']
				images = images.to(device=device, dtype=torch.float)
				target = target.to(device=device, dtype=torch.float)
				output = model(images)
				loss = criterion(output, target.long())
				if loss == 0 or not torch.isfinite(loss):
					continue
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				epoch_loss.append(float(loss))
				progress_bar.set_description( "Epoch: {}/{}. Loss: {:.7f}.".format(epoch, max_epoch, loss))
				progress_bar.update(1)

			scheduler.step(np.mean(epoch_loss))
			accuracy = evaluate(val_loader, model, device)
			current_lr = optimizer.param_groups[0]['lr']
			writer.add_scalar('Training Loss', loss, epoch * len(train_loader) + i)
			writer.add_scalar('Training Learning rate', current_lr, epoch)
			writer.add_scalar('Training Accuracy', accuracy, epoch)

			print("[INFO] Epoch {}: Loss: {:.4f}, Accuracy: {:.4f}%".format(epoch, loss, accuracy))
			if (loss < best_loss):
				best_loss = loss
				is_best_loss = True
			if (accuracy > best_acc):
				best_acc = accuracy
				is_best_acc = True
			save_checkpoint({
				'epoch': epoch,
				'arch': arch,
				'image_size': image_size,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'nrof_classes': nrof_classes},
				saved_path=weight_save_dir, is_best_loss=is_best_loss, is_best_acc=is_best_acc)
			is_best_loss = False
			is_best_acc = False
			last_epoch = epoch
		writer.close()
		progress_bar.close()
	except KeyboardInterrupt:
		save_checkpoint({
			'epoch': last_epoch,
			'arch': arch,
			'image_size': image_size,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'nrof_classes': nrof_classes},
			saved_path=weight_save_dir, is_best_loss=False, is_best_acc=False)
		writer.close()
		progress_bar.close()