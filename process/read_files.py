import os
import csv
from const import *

path = '..\\data\\train.csv'

with open(path, 'w') as csvfile:
  writer = csv.writer(csvfile, delimiter=';')
  writer.writerow(['file', 'class'])
  print(AUDIO_DIR)
  for root, dirs, files in os.walk(AUDIO_DIR):
    for filename in files:
        if(filename.endswith('.wav')) :
            print(filename)
            classif = filename.split('-')[1]
            writer.writerow([os.path.join(root,filename).replace('\data','')[1:].replace('\\','/'), classif])
