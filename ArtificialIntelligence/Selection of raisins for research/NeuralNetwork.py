#CTM
from random import random
from math import exp
import json


file_1 = open('weight_neural_network.json', "r")
data = json.load(file_1)
file_1.close()

file_6 = open('besni_set.json', "r")
data_6 = json.load(file_6)
file_6.close()

file_3 = open('train_set.json', "r")
data_3 = json.load(file_3)
train_dataset = data_3["train_set.json"]
file_3.close()

w = {
    'w1': [random()-0.5 for _ in range(7)],
    'w2': [random()-0.5 for _ in range(7)],
    'w3': [random()-0.5 for _ in range(7)],
    'w4': [random()-0.5 for _ in range(7)],
    'w5': [random()-0.5 for _ in range(7)],

    'w6': [random()-0.5 for _ in range(5)],
    'w7': [random()-0.5 for _ in range(5)],


    'w10': [random()-0.5],
    'w20': [random()-0.5],
    'w30': [random()-0.5],
    'w40': [random()-0.5],
    'w50': [random()-0.5],

    'w60': [random()-0.5],
    'w70': [random()-0.5]
}

v = {
    'v1': [],
    'v2': [],
    'v3': [],
    'v4': [],
    'v5': []
}

y = {
    'y1': [],
    'y2': []
}
all_gradients = {}

error = {
    'error': []
}


def activate(weights, inputs):
    s = 0
    for i in range(len(weights)):
        s += weights[i] * inputs[i]
    return s


def sigmoid(s, beta):
    return 1.0 / (1.0 + exp(-s * beta))


def forward_propagate(w, row, beta):
    x = row
    b = [1]
    v['v1'] = activate(w['w1'], x[:-2])
    v['v1'] += activate(w['w10'], b)
    v['v1'] = sigmoid(v['v1'], beta)

    v['v2'] = activate(w['w2'], x[:-2])
    v['v2'] += activate(w['w20'], b)
    v['v2'] = sigmoid(v['v2'], beta)

    v['v3'] = activate(w['w3'], x[:-2])
    v['v3'] += activate(w['w30'], b)
    v['v3'] = sigmoid(v['v3'], beta)

    v['v4'] = activate(w['w4'], x[:-2])
    v['v4'] += activate(w['w40'], b)
    v['v4'] = sigmoid(v['v4'], beta)

    v['v5'] = activate(w['w5'], x[:-2])
    v['v5'] += activate(w['w50'], b)
    v['v5'] = sigmoid(v['v5'], beta)

    y['y1'] = activate(w['w6'], [v['v1'], v['v2'], v['v3'],v['v4'], v['v5']])
    y['y1'] += activate(w['w60'], b)
    y['y1'] = sigmoid(y['y1'], beta)

    y['y2'] = activate(w['w7'],[v['v1'], v['v2'], v['v3'], v['v4'], v['v5']])
    y['y2'] += activate(w['w70'], b)
    y['y2'] = sigmoid(y['y2'], beta)

    y_ = [y['y1'], y['y2']]
    v_ = [v['v1'],v['v2'],v['v3'],v['v4'],v['v5']]
    return v_, y_


def sigmoid_derivative(f_s, beta):
    return beta * f_s * (1.0 - f_s)


def backward_propagate(d, beta, x,v,y):

    e = 0.0
    for j in range(len(y)):
        e += (d[j]-y[j]) ** 2
        gradient_vy_list = []
        gradient_vy_bias = []
        for k in v:
            gradient_vy_list.append(-(d[j]-y[j])*sigmoid_derivative(y[j], beta) * k)
        gradient_vy_bias.append(-(d[j]-y[j])*sigmoid_derivative(y[j], beta) * 1)
        all_gradients[f'g{j+6}'] = gradient_vy_list
        all_gradients[f'g{j*10+60}'] = gradient_vy_bias
    e = 0.5 * e
    e = round(e,4)


    weights_v_y = [w['w6'], w['w7']]

    for i in range(len(v)):
        gradient_xv_list = []
        gradient_xv_bias_list = []
        for j in range(len(x)):
            gradient_xv = 0
            for m in range(len(y)):
                gradient_xv += -(d[m]-y[m])*sigmoid_derivative(y[m], beta) * weights_v_y[m][i]
            gradient_xv = gradient_xv*sigmoid_derivative(v[i], beta)*x[j]
            gradient_xv_bias = gradient_xv*sigmoid_derivative(v[i], beta)*1
            gradient_xv_list.append(gradient_xv)
            gradient_xv_bias_list.append(gradient_xv_bias)
        all_gradients[f'g{i + 1}'] = gradient_xv_list
        all_gradients[f'g{i * 10 + 10}'] = gradient_xv_bias_list


def update_weight(w, gamma, all_gradients):
    weights_updat_vy = [w['w6'], w['w7']]
    weights_updat_bias_vy = [w['w60'], w['w70']]

    gradients_vy = [all_gradients['g6'], all_gradients['g7']]
    gradients_bias_vy = [all_gradients['g60'], all_gradients['g70']]

    for i in range(len(weights_updat_vy)):
        for j in range(len(weights_updat_vy[i])):
            weights_updat_vy[i][j] = weights_updat_vy[i][j] - gamma * gradients_vy[i][j]
            weights_updat_vy[i][j] = round(weights_updat_vy[i][j],4)

    for i in range(len(weights_updat_bias_vy)):
        for j in range(len(weights_updat_bias_vy[i])):
            weights_updat_bias_vy[i][j] = weights_updat_bias_vy[i][j] - gamma * gradients_bias_vy[i][j]
            weights_updat_bias_vy[i][j] = round(weights_updat_bias_vy[i][j], 4)

    weights_updat_xv = [w['w1'], w['w2'], w['w3'], w['w4'], w['w5']]
    weights_updat_bias_xv = [w['w10'], w['w20'], w['w30'], w['w40'], w['w50']]

    gradients_xv = [all_gradients['g1'], all_gradients['g2'], all_gradients['g3'], all_gradients['g4'], all_gradients['g5']]
    gradients_bias_xv = [all_gradients['g10'], all_gradients['g20'], all_gradients['g30'], all_gradients['g40'], all_gradients['g50']]

    for i in range(len(weights_updat_xv)):
        for j in range(len(weights_updat_xv[i])):
            weights_updat_xv[i][j] = weights_updat_xv[i][j] - gamma * gradients_xv[i][j]
            weights_updat_xv[i][j] = round(weights_updat_xv[i][j],4)

    for i in range(len(weights_updat_bias_xv)):
        for j in range(len(weights_updat_bias_xv[i])):
            weights_updat_bias_xv[i][j] = weights_updat_bias_xv[i][j] - gamma * gradients_bias_xv[i][j]
            weights_updat_bias_xv[i][j] = round(weights_updat_bias_xv[i][j],4)

def train_network(train_data, gamma, beta, e, iterations, number_epochs, value_of_difference):
    flag = True
    list_of_error_in_epoch = []
    while flag:
        list_y = []
        list_d = []
        iterations += 1
        for row in train_data:
            v, y = forward_propagate(w, row, beta)
            max_out = y.index(max(y))
            y_2 = [0, 0]
            y_2[max_out] = 1
            y_ = y_2
            list_y.append(y_)

            d = row[-2]
            list_d.append(d)
            # print('d', d)
            backward_propagate(d, beta, row[:-2], v, y)
            update_weight(w, gamma, all_gradients)
        print(f'Current epoch {iterations}')
        scores_of_error = 0
        if len(list_y) >= len(train_dataset):
            for i in range(len(list_y[-700:])):
                if list_y[-700 + i] != list_d[-700 + i]:
                    scores_of_error += 1
        print('scores_of_error', scores_of_error)
        list_of_error_in_epoch.append(scores_of_error)

        significant_change = [0]
        if len(list_of_error_in_epoch) >= number_epochs:
            for i in range(len(list_of_error_in_epoch[:-1])):
                if abs(list_of_error_in_epoch[i] - list_of_error_in_epoch[i+1]) >= value_of_difference:
                    significant_change.append(1)
            if sum(significant_change) == 0:
                data["weights"] = w
                flag = False
                break
            else:
                del list_of_error_in_epoch[0]


iterations = 0
gamma = 0.7
beta = 0.5
number_epochs = 25
value_of_difference = 6



def testing_neural_network(w, row, beta, score, besni_set):
    d = row[-2]
    v, y = forward_propagate(w, row, beta)
    max_out = y.index(max(y))
    y_2 = [0, 0]
    y_2[max_out] = 1

    d_3 = 0
    y_3 = 0
    if y_2 == [1, 0]:
        y_3 = 'Kecimen'
    elif y_2 == [0, 1]:
        y_3 = 'Besni'
        besni_set.append(row)
    if d == [1, 0]:
        d_3 = 'Kecimen'
    elif d == [0, 1]:
        d_3 = 'Besni'

    print('Expected output:', d_3)
    print('Neural network output: ', y_3)
    y = y_2
    if y == d:
        score.append(score[-1]+1)
        print('--->classified correctly')
    else:
        print('--->classified incorrectly')



answer = "train"
answer1 = '1'

if answer =='train':
    train_network(train_dataset, gamma, beta, error, iterations, number_epochs, value_of_difference)

besni_set = []

if answer == 'test':
    test_dataset = []
    if answer1 == '1':
        file_4 = open('test_set.json', "r")
        data_4 = json.load(file_4)
        test_dataset = data_4["test_set.json"]
        file_4.close()

    score = [0]

    for i in range(len(test_dataset)):
        print('Number set: ', i+1)
        testing_neural_network(data["weights"], test_dataset[i], beta, score, besni_set)
    print('Number of correctly classified sets: ',score[-1])
    data_6["besni_set"] = besni_set


g = open("weight_neural_network.json", "w")
json.dump(data, g)
g.close()

g2 = open("besni_set.json", "w")
json.dump(data_6, g2)
g2.close()
