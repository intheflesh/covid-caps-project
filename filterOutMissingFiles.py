import os
import pandas as pd

# since there are some misssing files in the orignal 'Data_Entry_2017.csv" file, we create a new one based on the folder with the actual images
inFileCSV = open('Data_Entry_2017.csv',"r")             # loading the original csv file
outFileCSV = open('Data_Entry_2017_Updated.csv',"w")    # creating the final csv file
pathOfFiles = r'D:\Data\covid-caps-backup\covid-caps_16.05.20\database_preprocessed'    # the folder where we keep the images
files = os.listdir(pathOfFiles)
first = inFileCSV.readline()
outFileCSV.write(first)
for line in inFileCSV:
    words = line.split(",")
    fileName = words[0]
    if fileName in files:
        outFileCSV.write(line)
