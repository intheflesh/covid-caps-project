import os
import numpy as np
# this code stacks the feature and label files created by the script:
# xray14_selectionMT.py


def initialSplit(folderPathIn,folderPathOut,N):
    listOfFiles = os.listdir(folderPathIn)
    listOfTuples = []
    for file in listOfFiles:
        index = file.split("_")[-1]
        if file.startswith("X"):
            tmpTuple = (file,"Y_labels_"+index)
            listOfTuples.append(tmpTuple)

    counter = 0
    afterSave = False
    for file in listOfTuples:
        pathOfData = os.path.join(folderPathIn,file[0])
        pathOfLabels = os.path.join(folderPathIn,file[1])
        if counter == 0 or afterSave == True:
            data = np.load(pathOfData)
            labels = np.load(pathOfLabels)
            afterSave = False
        else:
            data = np.concatenate((data, np.load(pathOfData)), axis=0)
            labels = np.concatenate((labels, np.load(pathOfLabels)), axis=0)
        if counter != 0 and (counter%N==0 or counter==len(listOfTuples)-1):
            np.save(os.path.join(folderPathOut,"data_"+str(counter)),data)
            np.save(os.path.join(folderPathOut,"labels_"+str(counter)),labels)
            afterSave = True
        counter += 1
        print(counter)



def mergeNpyFiles(folderPathIn,folderPathOut):
    listOfFiles = os.listdir(folderPathIn)
    listOfTuples = []
    for file in listOfFiles:
        index = file.split("_")[-1]
        if file.startswith("d"):
            tmpTuple = (file, "labels_" + index)
            listOfTuples.append(tmpTuple)
    counter = 0
    data = []
    labels = []
    for file in listOfTuples:
        pathOfData = os.path.join(folderPathIn, file[0])
        pathOfLabels = os.path.join(folderPathIn, file[1])
        if counter == 0:
            data = np.load(pathOfData)
            labels = np.load(pathOfLabels)
        else:
            data = np.concatenate((data, np.load(pathOfData)), axis=0)
            labels = np.concatenate((labels, np.load(pathOfLabels)), axis=0)
        counter+=1
    np.save(os.path.join(folderPathOut, "data"), data)
    np.save(os.path.join(folderPathOut, "labels"), labels)

# first create subsets of the original npy files, of size N (defaulted to be 90, so we would have 2 .npy files)

folderPathIn  = r'D:\Data\covid-caps-backup\covid-caps_16.05.20\npyFiles'               # a path to where we keep the .npy files created after xray14_seletionMT.py
folderPathOut  = r'D:\Data\covid-caps-backup\covid-caps_16.05.20\npyFilesConcatenated'  # intermediate out folder
N=90
initialSplit(folderPathIn,folderPathOut,N)

# now connect the 2 files we got
# we have to do this process as the straightforward concatenation takes too long.

folderPathOutFinal  = r'D:\Data\covid-caps-backup\covid-caps_16.05.20\npyFilesConcatenatedFinal'    # final out folder
mergeNpyFiles(folderPathOut,folderPathOutFinal)

# eventually we expect a file of about 13 giga, containing 2*90*500 = 90,000 images
# and the same amount of "one hot" label files with 5 classes:
# 1) No Findings
# 2) Tumors
# 3) Pleural Diseases
# 4) Lung Infection
# 5) Other