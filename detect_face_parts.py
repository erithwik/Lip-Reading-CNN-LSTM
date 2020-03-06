

#	Anish, you need to install imutils, its like pip install imutils lmao, also you need dlib, which also shouldn't be too hard to get
#	Also, you need to change line 59 to make it the best bounding box, like if you want more than just the lips
#	you need to adjust the x, x+w, y, and y+h, also you need to download the thing on line 18


from imutils import face_utils
import numpy as np
import glob
import pickle
import argparse
import imutils
import dlib
import cv2
import os
from os import listdir
from threading import Thread
from os.path import isfile, join

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
predictorDir = "C:\\Users\\anish\\Desktop\\SiemensStuff\\shape_predictor_68_face_landmarks.dat"
pathTofolder = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\LightDatasetResize\\LightDatasetResize'
outputPath = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\LightData_bound_lips\\'
bound_offset = 10

def createData(pathTofolder, output_path, fn):
	new_img_folder = os.path.splitext(fn)[0]
	new_path = output_path + new_img_folder
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	tmpFileName = os.path.join(pathTofolder, fn)
	images = glob.glob(tmpFileName + '\\*.jpg')
	images.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
	somefile = open(output_path + fn + '.txt', 'w')
	for item in images:
		somefile.write("%s\n" % item)
	somefile.close()
	temp_imgs = glob.glob(new_path + '\\*.jpg') 
	count = 0


# onlyfiles = [f for f in listdir(pathTofolder) if isfile(join(pathTofolder,f))]
# images = np.empty(len(onlyfiles), dtype=object)
# for n in range(0, len(images)):
# 	images[n] = cv2.imread(join(pathTofolder,onlyfiles[n]))

	for img in images:
		if count < 90:
			count+=1
			continue
		image = cv2.imread(img)
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(predictorDir)

	# load the input image, resize it, and convert it to grayscale
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
		rects = detector(gray, 1)

	# loop over the face detections
		for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				roi = image[y-bound_offset:y + h + bound_offset, x-bound_offset:x + w + bound_offset] #This needs to be changed later on...
				roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)


				cv2.imwrite(new_path + '\\image%d.jpg' % count, roi)
			# show the particular face part
			# 	cv2.imshow("ROI", roi)
			# #cv2.imshow("Image", clone)
			# 	cv2.waitKey(0)
				count+=1
				break;
	print ("Phonetic sound " + fn + " created")

	# visualize all facial landmarks with a transparent overlay
	# output = face_utils.visualize_facial_landmarks(image, shape)
	# cv2.imshow("Image", output)
	# cv2.waitKey(0)

def convImageMulti(pathTofolder, output_path, img, count, fn):
	new_img_folder = os.path.splitext(fn)[0]

	new_path = output_path + new_img_folder
	image = cv2.imread(img)
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictorDir)

	# load the input image, resize it, and convert it to grayscale
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y-bound_offset:y + h + bound_offset, x-bound_offset:x + w + bound_offset] #This needs to be changed later on...
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)


			cv2.imwrite(new_path + '\\image%d.jpg' % count, roi)
			# show the particular face part
			# 	cv2.imshow("ROI", roi)
			# #cv2.imshow("Image", clone)
			# 	cv2.waitKey(0)
			break;

def executeDataMulti(pathTofolder, output_path, img, count, fn):
	thread = Thread(target = convImageMulti, args = (pathTofolder, output_path, img, count, fn))
	return thread

def executeCreateData(pathTofolder, output_path, fn):
    thread = Thread(target = createData, args = (pathTofolder, output_path, fn))
    return thread

def main(pathTofolder, output_path):
    threads = [executeCreateData(pathTofolder, output_path, fn) for fn in os.listdir(pathTofolder)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
def main_mod(pathTofolder, output_path):
	for fn in os.listdir(pathTofolder):
		tmpFileName = os.path.join(pathTofolder, fn)
		images = glob.glob(tmpFileName + '\\*.jpg')
		images.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
		threads = [executeDataMulti(pathTofolder, output_path, img, count, fn) for count, img in enumerate(images, start=1)]
		for t in threads:
			t.start()
			print('new thread created!')
		for t in threads:
			t.join()
		print ("Phonetic sound " + fn + " created")

if __name__ == '__main__':
	# main_mod(pathTofolder=pathTofolder, output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp1', output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp2', output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp3', output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp4', output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp5', output_path=outputPath)
	main(pathTofolder=pathTofolder + '\\tmp6', output_path=outputPath)

