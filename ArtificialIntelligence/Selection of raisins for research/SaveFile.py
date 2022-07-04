# TTM
import csv
import json
import random
import numpy as np
from sklearn import preprocessing

file_1 = open('train_set.json', "r")
data1 = json.load(file_1)
file_1.close()

file_2 = open('test_set.json', "r")
data2 = json.load(file_2)
file_2.close()

file_3 = open('unnormed.json', "r")
data3 = json.load(file_3)
file_3.close()

parameters_list = []
exemplary_outputs = []
list_of_id = []
list_to_json3 = []
with open("Raisin_Dataset.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    id = 0
    for row in csv_reader:
        if id == 0:
            id += 1
        else:
            row[0] = row[0]
            row[1] = row[1].replace(',', '.')
            row[2] = row[2].replace(',', '.')
            row[3] = row[3].replace(',', '.')
            row[4] = row[4]
            row[5] = row[5].replace(',', '.')
            row[6] = row[6].replace(',', '.')
            if row[7] == 'Kecimen':
                row[7] = [1, 0]
            if row[7] == 'Besni':
                row[7] = [0, 1]
            parameters_list.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])])
            exemplary_outputs.append(row[7])
            list_of_id.append(id)
            list_to_json3.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), row[7], id])
            id += 1



x = np.array(parameters_list)
z = preprocessing.normalize(x, norm='max', axis=0)
x_normed = z


parameters_list2 = []
k = 0
for i in x_normed:
    auxiliary_list = []
    for j in i:
        auxiliary_list.append(j)
    auxiliary_list.append(exemplary_outputs[k])
    auxiliary_list.append(list_of_id[k])
    parameters_list2.append(auxiliary_list)
    k += 1


random.shuffle(parameters_list2)

list_to_json1 = []
list_to_json2 = []
counter = 0
for i in range(len(parameters_list2)):
    if counter < 700:
        list_to_json1.append(parameters_list2[i])
        counter += 1
    else:
        list_to_json2.append(parameters_list2[i])


data1["train_set.json"] = list_to_json1
data2["test_set.json"] = list_to_json2
data3["unnormed.json"] = list_to_json3

g = open("train_set.json", "w")
json.dump(data1, g)
g.close()

g = open("test_set.json", "w")
json.dump(data2, g)
g.close()

g = open("unnormed.json", "w")
json.dump(data3, g)
g.close()