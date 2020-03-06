import numpy as np

phonemes = [['f', 'v'],['r', 'q', 'w', 'u'],['b', 'p', 'm'],['th'],['ch','sh'],['o', 'oo'],['s', 'z', 'x'],['a', 'e', 'i', 'y'],['d', 'l', 'n', 't', 'j'],['g', 'k', 'h']]

def letterCompare(actualWord, arrayOutput):
	actualWord = list(actualWord)
	outputSize = len(arrayOutput)	#25
	wordSize = len(actualWord)		#5
	div = int(outputSize/wordSize)		#25/5 = 5

	changedWord = []

	for i in range(wordSize):		#5
		word = actualWord[i]			#b e g i n
		for j in range(div):		#5
			changedWord.append(word)# b b b b b e e e e e g g g g g i i i i i n n n n n

	while len(changedWord) < len(arrayOutput):
		changedWord.append(arrayOutput[len(arrayOutput) - 1]) # b b b b b e e e e e g g g g g i i i i i n n n n n n 

############################################### Non Phoneme Based
	nonAnswer = []

	for i in range(len(changedWord)):
		if changedWord[i] == arrayOutput[i]:
			nonAnswer.append(1)
		else:
			nonAnswer.append(0)

	nonAnswer = np.asarray(nonAnswer)
	nonAnswerMean = np.mean(nonAnswer)
############################################### Phoneme Based
	arrayOfPhonemed = []

	for i in range(len(changedWord)):
		for j in range(len(phonemes)):
			for k in range(len(phonemes[j])):
				if changedWord[i] == phonemes[j][k]:
					arrayOfPhonemed.append(phonemes[j])
					break

	actualArrayOfPhonemed = []

	for i in range(len(arrayOutput)):
		for j in range(len(phonemes)):
			for k in range(len(phonemes[j])):
				if arrayOutput[i] == phonemes[j][k]:
					actualArrayOfPhonemed.append(phonemes[j])
					break

	finalArray = []
	print(len(actualArrayOfPhonemed))
	print(len(arrayOfPhonemed))
	for i in range(len(arrayOfPhonemed)):
		if arrayOfPhonemed[i] == actualArrayOfPhonemed[i]:
			finalArray.append(1)
		else:
			finalArray.append(0)

	array2 = np.asarray(finalArray)

	answer = np.mean(array2)

######################################################
	
	return answer, nonAnswerMean

# output = ['p','p','p','p','k','k','k','e','g','g','g','g','i','i','i','i','n','n','n','n']
# actual = 'begin'
# # print(letterCompare(actual, output))
# answer, nonAnswer = letterCompare(actual, output)
# print("Phonemed Score: " + str(answer))
# print("Non Phonemed Score: " + str(nonAnswer))


