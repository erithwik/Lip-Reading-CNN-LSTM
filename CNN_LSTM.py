from keras.models import Model, load_model
from keras.layers import LSTM, Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
import numpy as np
import preprocessData
import math
import os
import keras
from keras.models import Sequential
from keras import backend as K

# creates a custom f1 score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


######## NOTE #############
# When trainging, remember to change the snapshot file name each time to match the date
snapshot_prefix = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\snapshots\\CNN_LSTM\\final_snapshots\\set16_report_face\\'
# snapshot_filename = 'model_09172017_trained_snapshot_v59_bound_lips_fixed_no_drop_trimmed_data_t6_fixed_normalized_same_lighting.h5'

# set model parameters

# splitpoint_a = math.floor(TotalData.shape[0]*0.7)
# splitpoint_b = math.floor(TotalData.shape[0]*0.9)
# data = TotalData[0:splitpoint_a]
# labels = TotalLabels[0:splitpoint_a]
# ValidationData = TotalData[splitpoint_a:splitpoint_b]
# ValidationLabels = TotalLabels[splitpoint_a:splitpoint_b]
# TestData = TotalData[splitpoint_b:]
# TestLabels = TotalLabels[splitpoint_b:]

# x_train = x_dat[0]
# x_val = x_dat[1]
# x_test = x_dat[2]

# y_train = y_dat[0]
# y_train = y_dat[1]
# y_test = y_dat[2]

# print(data.shape)
# data.astype('uint8')
# data = data.astype('uint8')
# ValidationData = ValidationData.astype('uint8')
# data = data / np.max(data)
# ValidationData = ValidationData / np.max(ValidationData)
# TestData = TestData.astype('uint8')
# TestData = TestData/np.max(TestData)

# x_train = x_train.astype('uint8')
# x_val = x_val.astype('uint8')
# x_test = x_test.astype('uint8')

# x_train = x_train/np.max(x_train)
# x_val = x_val/np.max(x_val)
# x_test = x_test/np.max(x_test)
def trainMultiModels():
	vn = 192
	conv_1_size = [32]
	conv_2_size = [64]

	imageSize = [64]
	timeStepSize = [15]

	lstmSize = [40]
	batchSize = [20]
	epochSize = [60]

	for i in imageSize:
		for t in timeStepSize:
			for b in batchSize:
				for e in epochSize:
					for c_1 in conv_1_size:
						for c_2 in conv_2_size:
							for l in lstmSize:
								runNN(imageSize = i, timeStepSize = t, batchSize = b, epochSize = e, versionNumber = vn, conv_1_size = c_1, conv_2_size = c_2, lstm_size = l)
								vn +=1
	return 1


def runNN(imageSize, timeStepSize, batchSize, epochSize, versionNumber, conv_1_size, conv_2_size, lstm_size):
	snapshot_info = 'v' + str(versionNumber) + '_image' + str(imageSize) + '_' + 'time' + str(timeStepSize) + '_' + 'batch' + str(batchSize) + '_' + 'epoch' + str(epochSize)
	snapshot_filename = 'model_09242017_trained_snapshot_'  + snapshot_info
	# define model architecture parameters
	numlabels = 213
	conv1 = conv_1_size
	conv2 = conv_2_size
	# conv3 = 128
	kernel_1 = 3
	kernel_2 = 3
#	kernel_3 = 3
	pool_1 = 2
	pool_2 = 2
#	pool_3 = 2
	lstm_1 = lstm_size
	# lstm_2 = timeStepSize
	drop_1 = 0.25
	drop_2 = 0.5
	# drop1 = .25
	# drop2 = .5
	# hidden1 = 256

# image dimensions
	imageHeight = imageSize
	imageLength = imageSize
	imageDepth = 3

	inputShape = (timeStepSize, imageHeight, imageLength, imageDepth)

# number of character labels


# preprocess and format Data
	x_train,y_train,x_val,y_val,x_test,y_test = preprocessData.formatData_time(time_step=timeStepSize, img_size=imageSize)


# create a tensorboard callback file and log directory
	tensorboard_path_dir = snapshot_prefix + 'logs_' + snapshot_info
	if not os.path.exists(tensorboard_path_dir):
		os.makedirs(tensorboard_path_dir)

	tbCallBack = keras.callbacks.TensorBoard(log_dir= tensorboard_path_dir, histogram_freq=10, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# setup convolutionalLSTM architecture
	model = Sequential()
	model.add(TimeDistributed(Conv2D(conv1, (kernel_1, kernel_1), padding='valid'),input_shape=inputShape))
	model.add(Activation('relu'))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(pool_1, pool_1))))
	model.add(Dropout(drop_1))
	print (model.output_shape)
	model.add(TimeDistributed(Conv2D(conv2, (kernel_2, kernel_2), padding='valid')))
	model.add(Activation('relu'))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(pool_2, pool_2))))
	model.add(Dropout(drop_2))
	# print (model.output_shape)
	# model.add(TimeDistributed(Conv2D(conv3, (kernel_3, kernel_3), padding='valid')))
	# model.add(Activation('relu'))
	# model.add(TimeDistributed(MaxPooling2D(pool_size=(pool_3, pool_3))))
	# print(model.output_shape)
# model.add(TimeDistributed(Conv2D(128, (3, 3), padding='valid')))
# model.add(Activation('relu'))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
# model.add(TimeDistributed(Conv2D(256, (3, 3), padding='valid')))
# model.add(Activation('relu'))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
# print(model.output_shape)

	model.add(TimeDistributed(Flatten()))
	print (model.output_shape)
	model.add(LSTM(units=lstm_1, return_sequences=True))
	print (model.output_shape)
	# # model.add(Dropout(0.5))
	# model.add(LSTM(units=lstm_2, return_sequences=True))
	# print (model.output_shape)

	model.add(TimeDistributed(Dense(numlabels)))
	print (model.output_shape)
	model.add(Activation('softmax'))
	print (model.output_shape)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=batchSize, epochs=epochSize, verbose=1, validation_data=(x_val, y_val), callbacks = [tbCallBack])

	score = model.evaluate(x_test, y_test, verbose=0)

	scoreTrain = model.evaluate(x_train, y_train, verbose=0) 

	scoreCV = model.evaluate(x_val,  y_val, verbose=0) 

	print('Train loss:', scoreTrain[0]) 
	print('Train accuracy:', scoreTrain[1])

	print('Val loss:', scoreCV[0]) 
	print('Val accuracy:', scoreCV[1])  

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


# saves keras model, with architecture, weights, model fitting configuration, etc, as a .h5 file
	model.save(snapshot_prefix + snapshot_filename + '.h5')

	snapshot_text = open(snapshot_prefix + snapshot_filename + '.txt', 'w')
	snapshot_text.write(snapshot_filename + ' information:\n')
	snapshot_text.write('Image Size = ' + str(imageHeight) + 'x' + str(imageLength) + 'x' + str(imageDepth) + '\n')
	snapshot_text.write('Time Step Size = ' + str(timeStepSize) + '\n')
	snapshot_text.write('Batch Size = ' + str(batchSize) + '\n')
	snapshot_text.write('Epoch Size = ' + str(epochSize) + '\n')
	snapshot_text.write('\n')
	snapshot_text.write('Model Architecture: \n')
	snapshot_text.write('conv1 = ' + str(conv1) + '\n')
	snapshot_text.write('conv2 = ' + str(conv2) + '\n')
	snapshot_text.write('pool1 = ' + str(pool_1) + '\n')
	snapshot_text.write('pool2 = ' + str(pool_2) + '\n')
	snapshot_text.write('kernel_1 = ' + str(kernel_1) + '\n')
	snapshot_text.write('kernel_2 = ' + str(kernel_2) + '\n')
	snapshot_text.write('lstm_1 = ' + str(lstm_1) + '\n')
	# snapshot_text.write('lstm_2 = ' + str(lstm_2) + '\n')
	snapshot_text.write('drop_1 = ' + str(drop_1) + '\n')
	snapshot_text.write('drop_2 = ' + str(drop_2) + '\n')
	snapshot_text.write('\n')
	snapshot_text.write('Model Training Results: \n')
	snapshot_text.write('Train Loss = ' + str(scoreTrain[0]) + '\n')
	snapshot_text.write('Train Accuracy = ' + str(scoreTrain[1]) + '\n')
	snapshot_text.write('Val Loss = ' + str(scoreCV[0]) + '\n')
	snapshot_text.write('Val Accuracy = ' + str(scoreCV[1]) + '\n')
	snapshot_text.write('Test Loss = ' + str(score[0]) + '\n')
	snapshot_text.write('Test Accuracy = ' + str(score[1]) + '\n')


if __name__ == '__main__':
    trainMultiModels()