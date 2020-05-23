import os
import pandas as pd
inFileCSV = open('Data_Entry_2017.csv',"r") #loading the csv file
outFileCSV = open('Data_Entry_2017_Updated.csv',"w") #loading the csv file
pathOfFiles = r'D:\Data\covid-caps-backup\covid-caps_16.05.20\database_preprocessed'
files = os.listdir(pathOfFiles)
first = inFileCSV.readline()
outFileCSV.write(first)
for line in inFileCSV:
    words = line.split(",")
    fileName = words[0]
    if fileName in files:
        outFileCSV.write(line)
