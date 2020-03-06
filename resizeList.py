import sys
import os.path
import glob

def addFrames(img, timesteps):
	num = len(img)
	if num % timesteps != 0:
		n = num % timesteps
		copy = img[num-(timesteps-n):]
		# print(len(copy))
		# print(num%timesteps)
		count = 0
		for i in copy:
			img.append(copy[count])
			count+=1
		# print(len(img)%timesteps)



	if len(img) % timesteps != 0:
		print("List resizing did not work!")
	return img

