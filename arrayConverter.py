import numpy as np
import math
from collections import Counter
from string import ascii_lowercase



def convCharArray(charArray):
	NNcount = []
	characterList = []
	correctCharList = []

	# the code for getting the array of characters of the CNN into NNoutput
	NNoutput = charArray # this is and

	omega = .15

	for character in ascii_lowercase:
		characterList.append(character)

	for character in ascii_lowercase:
		delta = NNoutput.count(character)
		NNcount.append(delta)

	for i in range(len(NNcount)):
		if(NNcount[i] >= 5):
			correctCharList.append(characterList[i])
		
	deleterList = []

	for i in range(len(NNoutput)):
		delta = False
		for j in range(len(correctCharList)):
			if(NNoutput[i] is correctCharList[j]):
				delta = True
		if delta == False:
			deleterList.append(i)

	for i in range(len(deleterList)):
		del NNoutput[deleterList[i]]
		for i in range(len(deleterList)):
			deleterList[i] -= 1

	finalResult = []

	for i in range(len(NNoutput)):
		if(i == 0):
			finalResult.append(NNoutput[i])
		else:
			if(NNoutput[i] != NNoutput[i-1]):
				finalResult.append(NNoutput[i])

	print (finalResult)

	words = open('dictionary_words_test.txt').read().split()
	for w in words:
		if finalResult == w:
			finalResult = w
			break

	return finalResult