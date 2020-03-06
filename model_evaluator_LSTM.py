from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from resizeList import addFrames
import cv2
import glob
from letterByLetter2 import letterCompare
#import pickle
import os
import letter_label_pairs
import histConverterV2
import numpy as np


# time_step = 0
# frame_size = 0
numlabels = 30
# frame_size_tuple = (frame_size, frame_size)

# face_cascade = cv2.CascadeClassifier('C:\\Users\\anish\\Desktop\\SiemensStuff\\haarcascade_frontalface_default.xml')
path_to_snapshot_folder = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\snapshots\\CNN_LSTM\\final_snapshots\\set8\\'
path_to_sample_folder_128 = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\single_evaluation_data_64\\bound_lips_64\\TestSetStanford_v3_128'
path_to_sample_folder_64 = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\single_evaluation_data_64\\bound_lips_64\\TestSetStanford_v2_64'
# snapshot_model = 'model_09182017_trained_snapshot_v83_image64_time20_batch60_epoch80.h5'
# model = load_model(path_to_snapshot_folder + snapshot_model)


model_list = glob.glob(path_to_snapshot_folder + '*.h5')
model_list.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))

for model_name in model_list:
	model = load_model(model_name)
	# load model paramaters like frame_size and time_step_size
	
	config = model.get_config()
	time_step = config[0]['config']['batch_input_shape'][1]
	frame_size = config[0]['config']['batch_input_shape'][2]
	path_to_sample = ''
	if frame_size == 64:
		path_to_sample = path_to_sample_folder_64
	elif frame_size == 128:
		path_to_sample = path_to_sample_folder_128

	for fn in os.listdir(path_to_sample):
		img = glob.glob(path_to_sample + '\\' + fn + '\\*.jpg')
		img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
		img = addFrames(img = img, timesteps = time_step)
		numFrames = len(img)
		tmp_data = np.zeros((int(numFrames/time_step), time_step, frame_size, frame_size, 3))

		d5_count = 0
		count = 0
		for image in img:
			frame = cv2.imread(image)
			tmp_data[d5_count, int(count%(time_step))] = frame
			if (count+1)%(time_step) == 0:
				d5_count+=1
			count += 1

		evaluation_sample = tmp_data
		evaluation_sample = evaluation_sample.astype('uint8')
		evaluation_sample = evaluation_sample/np.uint8(255)

		batch_size = len(img)

		evaluation_output = model.predict(x=evaluation_sample, batch_size = batch_size, verbose=0)


		print(evaluation_output.shape)
		evaluation_output = evaluation_output.reshape((evaluation_output.shape[0]*evaluation_output.shape[1], numlabels))
		# np.savetxt('C:\\Users\\anish\\Desktop\\SiemensStuff\\raw_prediction.txt', evaluation_output, fmt='%f')
		print(evaluation_output.shape)
		max_indices = np.argmax(evaluation_output, axis = 1)
		evaluation_output_conv = np.zeros_like(evaluation_output)
		evaluation_output_conv[np.arange(len(evaluation_output)), max_indices] = 1

# max_indices = np.argmax(evaluation_output, axis = 1)
# for i in evaluation_output:
# 	evaluation_output[i]
# 	evaluation_output[i, maxindeces = ()]
		output_conv_array = np.arange(len(letter_label_pairs.Label))+1 #this array is used to convert the model output predictions into the enum values that are stored in leter_label_pairs
		number_output = np.dot(evaluation_output_conv, output_conv_array)

		letter_output = []

		iterator = 0
		for i in number_output:
			for l in letter_label_pairs.Label:
				if l.value == number_output[iterator]:
					letter_output.append(l.name)
			iterator = iterator + 1
		output_file_acc = open(os.path.splitext(model_name)[0] + '.txt', 'a')
		output_file_temp = open(path_to_snapshot_folder + 'output_results_' + fn + '.txt', 'a')
		output_file_conv = open(path_to_snapshot_folder + 'output_results_' + fn + 'convChar.txt', 'a')
		output_file_temp.write(os.path.basename(model_name) + ': \n')
		output_file_conv.write(os.path.basename(model_name) + ': \n')
		count = 0
		for i in letter_output:
			if i == 'space':
				letter_output[count] = ' '
			output_file_temp.write(str(letter_output[count]))
			count+=1
		# determine accuracy of word classifications
		letter_output_string = ''.join(letter_output)
		print(letter_output_string)
		answer, nonAnswerMean = letterCompare(fn, letter_output)
		output_file_acc.write('\n')
		output_file_acc.write(fn + ': ' + str(answer) + ' ,  ' + str(nonAnswerMean))
		output_file_acc.write('\n')




		output_file_temp.write('\n')
		output_file_temp.write('\n')
		# output_file_conv.write('\n')
		output_file_temp.close()
		print(letter_output)
		final_result = histConverterV2.convCharArray(letter_output)
		count = 0
		for i in final_result:
			output_file_conv.write(str(final_result[count]))
			count +=1 
		output_file_conv.write('\n')
		output_file_conv.write('\n')
		output_file_conv.close()
		


	# output_file_temp.write('\n')
	# output_file_temp.write('\n')
	# output_file_temp.write(fn + ': \n')
	# for line in final_result:
	# 	output_file_temp.write(str(line))
	
		# print(final_result)


