import cv2
import glob
import pickle
import os
import letter_label_pairsv2
import math
import time
from resizeList import addFrames
import numpy as np

rootdir = 'C:\\Users\\anish\\Desktop\\SiemensStuff'
training_data_dir = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\bound_face_64x64\\train_data_bound_face_v1'

# time_step = 6
# numlabels = 30
# img_size = 128
img_dept = 3
num_training_items = 10755

def getNumTrainingItems(time_step):
    trainingItemCount = 0
    for fn in os.listdir(training_data_dir):
        if fn == 'labels_combined.txt':
            continue
        if fn == 'tmp_excluded_values':
            continue
        tmpFileName = os.path.join(training_data_dir, fn)
        img = glob.glob(tmpFileName + '\\*.jpg')
        if len(img)%time_step != 0:
            n = len(img) % time_step
            trainingItemCount += len(img) + (time_step-n)
        else:
            trainingItemCount+= len(img)
    return trainingItemCount


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    
# this function formats the samples, putting it all into one array
def formatSamples():
    Data = np.zeros((num_training_items, img_size, img_size, 3))
    # newData = []
    
    count = 0
    for fn in os.listdir(training_data_dir):
        if fn == 'labels_combined.txt':
            continue
        tmpFileName = os.path.join(training_data_dir, fn)
        img = glob.glob(tmpFileName + '\\*.jpg')

        img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
        for fname in img:    
            x = cv2.imread(fname)
##            newx,newy = x.shape[1]/6, x.shape[0]/6
##            x = cv2.resize(x,(int(newx), int(newy)))
            # newData.append(x)
            Data[count] = x
##            cv2.imwrite(fname, x)
            count += 1

        print ('Phonetic sound ' + fn + ' formatted')

    # Data  = np.array(newData)
    print(Data.shape)
    return Data




def formatData_time(time_step, img_size):
    if img_size == 128:
        training_data_dir = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\bound_lips_128\\bound_lips_v17_128'
    elif img_size == 64:
        training_data_dir = 'C:\\Users\\anish\\Desktop\\SiemensStuff\\train_data\\bound_face_64x64\\train_data_bound_face_v1'

    generateLabels(training_data_dir, time_step = time_step)
    num_training_items = getNumTrainingItems(time_step=time_step)
    num_time_items = num_training_items/time_step


    splitpoint_a = math.floor(0.7*num_time_items)
    splitpoint_b = math.floor(0.9*num_time_items)
    # total_data = np.zeros((num_time_items, time_step, img_size  img_size, 3))
    # x_train = np.zeros(splitpoint_a, time_step, img_size, img_size, 3)
    # x_val = np.zeros(splitpoint_b-splitpoint_a:splitpoint_b, time_step, img_size, img_size, 3)
    # x_test = np.zeros(num_time_items-splitpoint_b, time_step, img_size, img_size, 3)
    x_train = []
    x_val = []
    x_test = []

    y_train = []
    y_val = []
    y_test = []
    
    # newData = []
    d5_count = 0
    count = 0
    iterator = 0
    for fn in os.listdir(training_data_dir):
        if fn == 'labels_combined.txt':
            continue
        if fn == 'tmp_excluded_values':
            continue
        tmpFileName = os.path.join(training_data_dir, fn)
        img = glob.glob(tmpFileName + '\\*.jpg')
        img = addFrames(img = img, timesteps = time_step)
        img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
        num_time_items_img = int(len(img)/time_step)

        splitpoint_a_img = math.floor(0.7*num_time_items_img)
        splitpoint_b_img = math.floor(0.9*num_time_items_img)

        temp_Data = np.zeros((num_time_items_img, time_step, img_size, img_size, img_dept))
        d5_count_temp = 0
        count_temp = 0
        for fname in img:    
            x = cv2.imread(fname)
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
##             newx,newy = x.shape[1]/6, x.shape[0]/6
##             x = cv2.resize(x,(int(newx), int(newy)))
            # newData.append(x)
            temp_Data[d5_count_temp, int(count_temp%(time_step))] = x
            # total_data[d5_count, int(count%time_step)] = x
            if (count_temp+1)%(time_step) == 0:
                d5_count_temp+=1
                d5_count += 1
##              cv2.imwrite(fname, x)
            count_temp += 1
            count+=1

        temp_labels = getLabels_time_modified(fn = fn, img_list = img, time_step=time_step, training_data_dir = training_data_dir)
        print(temp_labels.shape)

        #normalizing data values
        temp_Data = temp_Data.astype('uint8')
        temp_Data = temp_Data/(np.max(temp_Data))


        if iterator == 0:
            x_train = temp_Data[0:splitpoint_a_img]
            x_val = temp_Data[splitpoint_a_img:splitpoint_b_img]
            x_test = temp_Data[splitpoint_b_img:]

            y_train = temp_labels[0:splitpoint_a_img]
            y_val = temp_labels[splitpoint_a_img:splitpoint_b_img]
            y_test = temp_labels[splitpoint_b_img:]  
        else:   
            x_train = np.concatenate((x_train, temp_Data[0:splitpoint_a_img]), axis=0)
            x_val = np.concatenate((x_val, temp_Data[splitpoint_a_img:splitpoint_b_img]), axis=0)
            x_test = np.concatenate((x_test, temp_Data[splitpoint_b_img:]), axis=0)

            y_train = np.concatenate((y_train, temp_labels[0:splitpoint_a_img]), axis=0)
            y_val = np.concatenate((y_val, temp_labels[splitpoint_a_img:splitpoint_b_img]), axis =0)
            y_test = np.concatenate((y_test, temp_labels[splitpoint_b_img:]), axis = 0)

            print(x_train.shape)
        print ('Phonetic sound ' + fn + ' formatted')
        iterator+=1

    # Data  = np.array(newData)
    # x_train = np.asarray(x_train)
    # x_val = np.asarray(x_val)
    # x_test = np.asarray(x_test)

    # y_train = np.asarray([y_train])
    # y_val = np.asarray([y_val])
    # y_test = np.asarray([y_test])
    print("X shapes: ")
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    print("Y shapes: ")
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)


    # shuffles the data and labels:
    shuffle_in_unison(x_train, y_train)
    shuffle_in_unison(x_val, y_val)
    shuffle_in_unison(x_test, y_test)



   # (x_train, x_val, x_test)
   #  y_total = (y_train, y_val, y_test)
    return (x_train, y_train, x_val, y_val, x_test, y_test)




def generateLabels(training_data_dir, time_step):
##    letter_label_pairs = open( 'C:\\Users\\anish\\Desktop\\SiemensStuff\\letter_label_pairs.txt', 'r')
##    pairs = np.chararray([29, 2]) 
##    for i, line in enumerate(letter_label_pairs):
##        str1 = line.split()
##        pairs[i, 0] = str1[0]
##        pairs[i, 1] = str1[1]

##    letter_label_pairs.close()
##    masterlabels = open(rootdir + '\\trainingdata\\labels_combined.txt', 'a')
    masterlabels = open(training_data_dir + '\\labels_combined.txt','w')
    masterlabels.seek(0)     #clear master labels file before working with it
    masterlabels.truncate(0)
    masterlabels.close()
    for fn in os.listdir(training_data_dir):
        if fn == 'labels_combined.txt' or fn == 'tmp_excluded_values':
            continue
        imgs = glob.glob(training_data_dir + '\\' + fn + "\\*.jpg")
        imgs = addFrames(img = imgs, timesteps = time_step)

        imgs.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
        labels = open(training_data_dir + '\\' + fn + '\\testlabels.txt','w')
        masterlabels = open(training_data_dir + '\\labels_combined.txt', 'a')
        for imgName in imgs:
            if fn == 'aa' or fn=='ah':
                labels.write('a\n')
            elif fn == 'eh':
                labels.write('e\n')
            elif fn == 'ii':
                labels.write('i\n')
            elif fn == 'oh':
                labels.write('o\n')
            elif fn == 'u' or fn == 'uh':
                labels.write('u\n')
            elif fn == 'blank':
                labels.write(' \n')
            else:    
                labels.write(fn + '\n')

        labels.close()
        labels = open(training_data_dir + '\\' + fn + '\\testlabels.txt','r') #have to reopen in read mode        
        masterlabels.write(labels.read());
        
        masterlabels.close()
        

def getLabels():
    fileList = glob.glob(training_data_dir + '\\**\\*.jpg')
    fileList.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
    lines = np.zeros((len(fileList), len(letter_label_pairsv2.Label)))
    print(lines.shape)
    f = open(training_data_dir + "\\labels_combined.txt", "r")
    iterator = 0
    for line in f:
        for l in letter_label_pairsv2.Label:
            if line.strip()==l.name and iterator<len(fileList):
                lines[iterator, (l.value)-1] = 1
                break
            if line.strip()==' ' and iterator<len(fileList):
                lines[iterator,  29]  = 1# this is hard coded for now - should try to automate later, for the [space] label

        iterator+=1
                
            
    return lines


def getLabels_time():
    fileList = glob.glob(training_data_dir + '\\**\\*.jpg')
    fileList.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
    lines = np.zeros((int(len(fileList)/(time_step)), time_step, len(letter_label_pairsv2.Label)))
    print(lines.shape)
    f = open(training_data_dir + "\\labels_combined.txt", "r")
    iterator = 0
    d5_count = 0
    for line in f:
        label_iterator = 0
        for l in letter_label_pairsv2.Label:
            if line.strip()==l.name and iterator<len(fileList):
                lines[d5_count, int(iterator%(time_step)), (l.value)-1] = 1
                label_iterator += 1
        if label_iterator == 0 and iterator<len(fileList):
                lines[d5_count, int(iterator%(time_step)), 29] = 1
        if (iterator+1)%(time_step) == 0:
            d5_count+=1
        iterator+=1
                

    return lines



def getLabels_time_modified(fn,img_list, time_step, training_data_dir):
    numlabels = 213
    # fileList = glob.glob(training_data_dir + '\\' + fn +'\\*.jpg')
    # fileList.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
    lines = np.zeros((int(len(img_list)/(time_step)), time_step, len(letter_label_pairsv2.Label)))
    f = open(training_data_dir + '\\' + fn + "\\testlabels.txt", "r")
    iterator = 0
    d5_count = 0
    for line in f:
        label_iterator = 0
        for l in letter_label_pairsv2.Label:
            if line.strip()==l.name and iterator<len(img_list):
                lines[d5_count, int(iterator%(time_step)), (l.value)-1] = 1
                label_iterator += 1
        if label_iterator == 0 and iterator<len(img_list):
                lines[d5_count, int(iterator%(time_step)), 29] = 1
        if (iterator+1)%(time_step) == 0:
            d5_count+=1
        iterator+=1

    return lines


# old code
def formatData(net_type="cnn"):
    if net_type == "cnn":
        samples = formatSamples()
        labels = getLabels()
    elif net_type == "cnn_lstm":
        samples = formatSamples_time()
        labels = getLabels_time()
        
    shuffle_in_unison(samples, labels)

    return samples, labels    



##data_test = formatData()
##print(data_test.shape)
##print(x.shape)
##
##print(Data.shape)
##print(Data[0,0,0].dtype.name)

if __name__ == '__main__':
    generateLabels()
    # print(getNumTrainingItems())