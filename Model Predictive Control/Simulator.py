# CTM
import tkinter
import random
from tkinter import *
from random import random
import json
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

file_1 = open('dane_historyczne.json', "r")
data = json.load(file_1)
data_historicall_symulator = data["dane"]
data_historicall_symulator_parametry = data["parametry"]
file_1.close()


def activate(weights, inputs):
    s = 0
    for i in range(len(weights)):
        s += weights[i] * inputs[i]
    return s


def linear(s):
    return s


def forward_propagate(weights, inputs):
    y = activate(weights, inputs)
    y = linear(y)
    return y


def linear_derivative():
    return 1


def backward_propagate(d1, d2, y1, y2, x_1n, x_2n, u_n):
    gradient_w01 = (-(d2 - y2) * linear_derivative() * 1)
    gradient_w02 = (-(d2 - y2) * linear_derivative() * 1)
    gradient_w1 = (-(d1 - y1) * linear_derivative() * x_1n)
    gradient_w2 = (-(d1 - y1) * linear_derivative() * x_2n)
    gradient_w3 = (-(d1 - y1) * linear_derivative() * u_n)
    gradient_w4 = (-(d2 - y2) * linear_derivative() * x_1n)
    gradient_w5 = (-(d2 - y2) * linear_derivative() * x_2n)
    gradient_w6 = (-(d2 - y2) * linear_derivative() * u_n)

    gradients = [gradient_w01, gradient_w02, gradient_w1, gradient_w2, gradient_w3, gradient_w4, gradient_w5,
                 gradient_w6]
    return gradients


def update_weight(we01, we02, we1, we2, we3, we4, we5, we6, gamma, d1, d2, y1, y2, x_1n, x_2n,
                  u_n):
    weights_updat = [we01, we02, we1, we2, we3, we4, we5, we6]

    gradients = backward_propagate(d1, d2, y1, y2, x_1n, x_2n, u_n)

    temporary_list = weights_updat
    for i in range(len(temporary_list)):
        temporary_list[i] = weights_updat[i] - gamma * gradients[i]
        weights_updat[i] = round(temporary_list[i], 4)

    we01 = weights_updat[0]
    we02 = weights_updat[1]
    we1 = weights_updat[2]
    we2 = weights_updat[3]
    we3 = weights_updat[4]
    we4 = weights_updat[5]
    we5 = weights_updat[6]
    we6 = weights_updat[7]

    return we01, we02, we1, we2, we3, we4, we5, we6


def control_device(we01, we02, we1, we2, we3, we4, we5, we6, x_1n, x_2n, x_1star, x_2star, u_n):
    u_n_star = -(
            (x_1n * we1 + x_2n * we2 + we01 - x_1star) * we3 + (x_1n * we4 + x_2n * we5 + we02 - x_2star) * we6) / (
                       we3 ** 2 + we6 ** 2)
    roznica = u_n - u_n_star
    u_nowe = u_n - 0.3 * roznica
    return u_nowe


def object(a11, a12, a21, a22, b11, b21, x_1n, x_2n, u_n):
    x_1np1 = a11 * x_1n + a12 * x_2n + b11 * u_n
    x_2np1 = a21 * x_1n + a22 * x_2n + b21 * u_n
    return x_1np1, x_2np1


def train_network(train_data, gamma, we01, we02, we1, we2, we3, we4, we5, we6):
    temporary_list = []
    for row in train_data:
        x_1n = row[0]
        x_2n = row[1]
        u_n = row[2]
        x_1np1 = row[3]
        x_2np1 = row[4]

        y1 = forward_propagate([we1, we2, we3, we01], [x_1n, x_2n, u_n, 1])
        y2 = forward_propagate([we4, we5, we6, we02], [x_1n, x_2n, u_n, 1])
        d1 = x_1np1
        d2 = x_2np1

        error1 = 0.5 * (d1 - y1) ** 2
        error2 = 0.5 * (d2 - y2) ** 2
        temporary_list.append(error1)
        temporary_list.append(error2)
        we01, we02, we1, we2, we3, we4, we5, we6 = update_weight(we01, we02, we1, we2, we3, we4, we5, we6, gamma, d1,
                                                                 d2, y1, y2, x_1n, x_2n, u_n)
    numerator = sum(temporary_list)
    divider = len(temporary_list)
    if divider != 0:
        error = numerator / divider
    else:
        error = 0

    return we01, we02, we1, we2, we3, we4, we5, we6, error


def run_mpc(we01, we02, we1, we2, we3, we4, we5, we6, x_01, x_02, lic_iteracji, lis_train_data, lis_iteracji,
            lis_bledow_error, order, a11, a12, a21, a22, b11, b21, param_gamma, x1star, x2star, hist_data, lis_iter_his,
            coun):
    flag = True
    order2 = True
    if hist_data[0] == False and coun == 1:
        print('x[1](0)', round(x_01, 4))
        print('x[2](0)', round(x_02, 4))
        u_1 = 0.3*-((x_01 * we1 + x_02 * we2 + we01 - x1star) * we3 + (x_01 * we4 + x_02 * we5 + we02 - x2star) * we6) / (
                we3 ** 2 + we6 ** 2)
        u_1 = round(u_1, 4)
        print('u_1', u_1)
        x_1_1, x_2_1 = object(a11, a12, a21, a22, b11, b21, x_01, x_02, u_1)
        x_1_1 = round(x_1_1, 4)
        x_2_1 = round(x_2_1, 4)
        print('x[1](n+1) | x[2](n+1)', x_1_1, '|', x_2_1)
        lis_train_data.append([x_01, x_02, u_1, x_1_1, x_2_1])
        wei01, wei02, wei1, wei2, wei3, wei4, wei5, wei6, error = train_network(lis_train_data, param_gamma, we01, we02,
                                                                                we1, we2, we3, we4, we5, we6)
        lis_bledow_error.append(error)

        lis_iteracji.append(1)
        lis_iter_his.append(1)

    if hist_data[0] == True and coun == 1:
        x_1_n = hist_data[1][-2]
        x_2_n = hist_data[1][-1]
        u_nm1 = hist_data[1][-3]
        lis_iteracji.append(1)

        weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6 = hist_data[2]

        u_n = control_device(weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6, x_1_n, x_2_n, x1star, x2star,
                             u_nm1)
        u_n = round(u_n, 4)
        print('u_n', u_n)
        x_1np1, x_2np1 = object(a11, a12, a21, a22, b11, b21, x_1_n, x_2_n, u_n)
        x_1np1 = round(x_1np1, 4)
        x_2np1 = round(x_2np1, 4)
        print('x[1](n+1) | x[2](n+1)', x_1np1, '|', x_2np1)
        lis_train_data.append([x_1_n, x_2_n, u_n, x_1np1, x_2np1])

        weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6, error = train_network(lis_train_data, param_gamma,
                                                                                        weig01, weig02, weig1, weig2,
                                                                                        weig3, weig4, weig5, weig6)
        lis_bledow_error.append(error)

        lis_iter_his.append(lis_iter_his[-1] + 1)

    if coun != 1:
        weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6 = we01, we02, we1, we2, we3, we4, we5, we6
    else:
        if hist_data[0] == True:
            weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6 = weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6
        else:
            weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6 = wei01, wei02, wei1, wei2, wei3, wei4, wei5, wei6
    while flag:

        x_1_n = lis_train_data[-1][-2]
        x_2_n = lis_train_data[-1][-1]
        u_nm1 = lis_train_data[-1][-3]

        u_n = control_device(weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6, x_1_n, x_2_n, x1star, x2star,
                             u_nm1)
        u_n = round(u_n, 4)
        print('u_n', u_n)
        x_1np1, x_2np1 = object(a11, a12, a21, a22, b11, b21, x_1_n, x_2_n, u_n)
        x_1np1 = round(x_1np1, 4)
        x_2np1 = round(x_2np1, 4)
        print('x[1](n+1) | x[2](n+1)', x_1np1, '|', x_2np1)
        lis_train_data.append([x_1_n, x_2_n, u_n, x_1np1, x_2np1])
        weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6, error = train_network(lis_train_data, param_gamma,
                                                                                        weig01, weig02, weig1, weig2,
                                                                                        weig3, weig4, weig5, weig6)
        lis_bledow_error.append(error)
        print('w01, w02, w1, w2, w3, w4, w5, w6!!!', weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6)

        lis_iteracji.append(lis_iteracji[-1] + 1)
        lis_iter_his.append(lis_iter_his[-1] + 1)

        if lis_iteracji[-1] == lic_iteracji:

            flag = False
            x_ostatnie1 = lis_train_data[-1][-2]
            x_ostatnie2 = lis_train_data[-1][-1]

            return lis_train_data, lis_iteracji, lis_bledow_error, weig01, weig02, weig1, weig2, weig3, weig4, weig5, weig6, x_ostatnie1, x_ostatnie2, lis_iter_his


def model_learning(lis_data, lis_error, lis_iter_his, weight01, weight02, weight1, weight2, weight3, weight4, weight5,
                   weight6, param_gamma):
    train_data_learning = []
    lis_iter_his.append(1)
    for i in range(len(lis_data)):

        x_1_n = lis_data[i][0]
        x_2_n = lis_data[i][1]
        u_n = lis_data[i][2]
        x_1np1 = lis_data[i][3]
        x_2np1 = lis_data[i][4]

        train_data_learning.append([x_1_n, x_2_n, u_n, x_1np1, x_2np1])
        weight01, weight02, weight1, weight2, weight3, weight4, weight5, weight6, error = train_network(
            train_data_learning, param_gamma, weight01, weight02, weight1, weight2, weight3, weight4, weight5, weight6)
        lis_error.append(error)
        lis_iter_his.append(lis_iter_his[-1] + 1)
    del lis_iter_his[-1]
    x_h0 = train_data_learning[-1][0]
    x_h1 = train_data_learning[-1][1]
    u_nh = train_data_learning[-1][2]
    x_1np1h = train_data_learning[-1][3]
    x_2np1h = train_data_learning[-1][4]

    return weight01, weight02, weight1, weight2, weight3, weight4, weight5, weight6, x_h0, x_h1, lis_error, lis_iter_his, u_nh, x_1np1h, x_2np1h


root = tkinter.Tk()
root.configure(bg="white")
root.title("MPC")
root.geometry("1600x1200")

bg = PhotoImage(file="schemat mpc8.png")

label1 = Label(root, image=bg)
label1.place(x=300, y=0)

fig1, axs1 = plt.subplots(1, figsize=(3.5, 2.5))
fig2, axs2 = plt.subplots(1, figsize=(3.5, 2.5))
fig3, axs3 = plt.subplots(1, figsize=(3.5, 2.5))

axs1.set_title('Wartość decyzji sterującej w poszczególnych iteracjach', fontsize=6)
axs1.set_xlabel('liczba iteracji', fontsize=6)
axs1.set_ylabel('wartość u(n)', fontsize=6)
canvas = FigureCanvasTkAgg(fig1, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=520, y=70)
fig1.tight_layout()

axs2.set_title('Wartość stanu obiektu w poszczególnych iteracjach', fontsize=6)
axs2.set_xlabel('liczba iteracji', fontsize=6)
axs2.set_ylabel('wartość x(n+1)', fontsize=6)

canvas = FigureCanvasTkAgg(fig2, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=1065, y=70)
fig2.tight_layout()

axs3.set_title('Wartość błędu e w poszczególnych iteracjach \n e = 0.5*(wzorcowe wyjście - wyjście sieci)**2',
               fontsize=6)
axs3.set_xlabel('liczba iteracji', fontsize=6)
axs3.set_ylabel('wartość błędu', fontsize=6)
canvas = FigureCanvasTkAgg(fig3, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=520, y=510)
fig3.tight_layout()

label_title0_0 = tkinter.Label(root, text="Wzór opisujący rzeczywisty obiekt sterowania: x(n+1) = A * x(n) + B * u(n)")
label_title0_0.grid(row=0, column=0, columnspan=3)

label_title1_0 = tkinter.Label(root, text="Czy chcesz wykorzystać dane historyczne do wstępnego 'wyuczenia' modelu:",
                               bg="white")
label_title1_0.grid(row=1, column=0, columnspan=3)


button2_0 = tkinter.Button(root, text="tak", bg="white", fg="black", width=15,
                           command=lambda: yes_his(war_zad_iter))
button2_0.grid(row=2, column=0)


button2_1 = tkinter.Button(root, text="nie", bg="white", width=15,
                           command=lambda: not_his())
button2_1.grid(row=2, column=1)


x_1star = DoubleVar()
x_2star = DoubleVar()
iterations = DoubleVar()
gamma = DoubleVar()

a_11 = DoubleVar()  # a(11)
a_12 = DoubleVar() # a(12)
a_21 = DoubleVar()  # a(21)
a_22 = DoubleVar()  # a(22)
b_11 = DoubleVar()  # b(11)
b_21 = DoubleVar()  # b(21)
x_01 = DoubleVar() # x[1](0)
x_02 = DoubleVar()  # x[2](0)

w01 = DoubleVar()  # w(01)
w02 = DoubleVar()  # w(02)
w1 = DoubleVar()  # w(1)
w2 = DoubleVar()  # w(2)
w3 = DoubleVar()  # w(3)
w4 = DoubleVar()  # w(4)
w5 = DoubleVar()  # w(5)
w6 = DoubleVar()  # w(6)

war_zad_iter = []


def yes_his(war_zad_iter):
    historical_data_yes(counter, lista_train_data, lista_iteracji,
                        lista_bledow_error, data_to_run, lis_iter_history,war_zad_iter)
    global text_input10_1
    global text_input11_1
    global text_input12_1
    global text_input13_1
    label_title10_0 = tkinter.Label(root, text=f"Wartość zadana x[1]*", bg="white")
    label_title10_0.grid(row=10, column=0)

    label_title11_0 = tkinter.Label(root, text=f"Wartość zadana x[2]*", bg="white")
    label_title11_0.grid(row=11, column=0)

    label_title12_0 = tkinter.Label(root, text=f"Liczba iteracji", bg="white")
    label_title12_0.grid(row=12, column=0)

    label_title13_0 = tkinter.Label(root, text="Parametr uczenia sieci:", bg="white")
    label_title13_0.grid(row=13, column=0)

    text_input10_1 = tkinter.Entry(root,  width=15)  # Wartość zadana x[1]* textvariable=x_1star,
    text_input10_1.grid(row=10, column=1)

    text_input11_1 = tkinter.Entry(root,  width=15)  # Wartość zadana x[2]* textvariable=x_2star,
    text_input11_1.grid(row=11, column=1)

    text_input12_1 = tkinter.Entry(root,  width=15)  # zliczba iteracji1 textvariable=iterations,
    text_input12_1.grid(row=12, column=1)

    text_input13_1 = tkinter.Entry(root,  width=15)  # gamma textvariable=gamma,
    text_input13_1.grid(row=13, column=1)


def not_his():
    historical_data_not(counter, lista_train_data, lista_iteracji,
                        lista_bledow_error, data_to_run, lis_iter_history,war_zad_iter)
    global text_input2_1  # a(11)
    global text_input3_1  # a(12)
    global text_input4_1  # a(21)
    global text_input5_1  # a(22)
    global text_input6_1  # b(11)
    global text_input7_1  # b(21)
    global text_input8_1  # x[1](0)
    global text_input9_1  # x[2](0)
    global text_input10_1  # Wartość zadana x[1]*
    global text_input11_1  # Wartość zadana x[2]*
    global text_input12_1  # liczba iteracji
    global text_input13_1  # gamma
    global text_input14_1  # w(01)
    global text_input15_1  # w(02)
    global text_input16_1  # w(1)
    global text_input17_1  # w(2)
    global text_input18_1  # w(3)
    global text_input19_1  # w(4)
    global text_input20_1  # w(5)
    global text_input21_1  # w(6)
    label_title1_0 = tkinter.Label(root, text="Podaj parametry obiektu:", bg="white")
    label_title1_0.grid(row=1, column=0)

    label_title2_0 = tkinter.Label(root, text="Parametr a(11):", bg="white")
    label_title2_0.grid(row=2, column=0)

    label_title3_0 = tkinter.Label(root, text="Parametr a(12):", bg="white")
    label_title3_0.grid(row=3, column=0)

    label_title4_0 = tkinter.Label(root, text="Parametr a(21):", bg="white")
    label_title4_0.grid(row=4, column=0)

    label_title5_0 = tkinter.Label(root, text="Parametr a(22):", bg="white")
    label_title5_0.grid(row=5, column=0)

    label_title6_0 = tkinter.Label(root, text="Parametr b(11):", bg="white")
    label_title6_0.grid(row=6, column=0)

    label_title7_0 = tkinter.Label(root, text="Parametr b(21):", bg="white")
    label_title7_0.grid(row=7, column=0)

    label_title8_0 = tkinter.Label(root, text="Wartość x[1](0):", bg="white")
    label_title8_0.grid(row=8, column=0)

    label_title9_0 = tkinter.Label(root, text="Wartość x[2](0):", bg="white")
    label_title9_0.grid(row=9, column=0)

    label_title10_0 = tkinter.Label(root, text=f"Wartość zadana x[1]*", bg="white")
    label_title10_0.grid(row=10, column=0)

    label_title11_0 = tkinter.Label(root, text=f"Wartość zadana x[2]*", bg="white")
    label_title11_0.grid(row=11, column=0)

    label_title12_0 = tkinter.Label(root, text=f"Liczba iteracji", bg="white")
    label_title12_0.grid(row=12, column=0)

    label_title13_0 = tkinter.Label(root, text="Parametr uczenia sieci:", bg="white")
    label_title13_0.grid(row=13, column=0)

    label_title14_0 = tkinter.Label(root, text="waga w(0)'", bg="white")
    label_title14_0.grid(row=14, column=0)

    label_title15_0 = tkinter.Label(root, text="waga w(0)''", bg="white")
    label_title15_0.grid(row=15, column=0)

    label_title16_0 = tkinter.Label(root, text="waga w(1)", bg="white")
    label_title16_0.grid(row=16, column=0)

    label_title17_0 = tkinter.Label(root, text="waga w(2)", bg="white")
    label_title17_0.grid(row=17, column=0)

    label_title18_0 = tkinter.Label(root, text="waga w(3)", bg="white")
    label_title18_0.grid(row=18, column=0)

    label_title19_0 = tkinter.Label(root, text="waga w(4)", bg="white")
    label_title19_0.grid(row=19, column=0)

    label_title20_0 = tkinter.Label(root, text="waga w(5)", bg="white")
    label_title20_0.grid(row=20, column=0)

    label_title21_0 = tkinter.Label(root, text="waga w(6)", bg="white")
    label_title21_0.grid(row=21, column=0)

    # pola do wpisywania tekstu
    text_input2_1 = tkinter.Entry(root, width=15)  # a(11)
    text_input2_1.grid(row=2, column=1)

    text_input3_1 = tkinter.Entry(root, width=15)  # a(12)
    text_input3_1.grid(row=3, column=1)

    text_input4_1 = tkinter.Entry(root, width=15)  # a(21)
    text_input4_1.grid(row=4, column=1)

    text_input5_1 = tkinter.Entry(root, width=15)  # a(22)
    text_input5_1.grid(row=5, column=1)

    text_input6_1 = tkinter.Entry(root, width=15)  # b(11)
    text_input6_1.grid(row=6, column=1)

    text_input7_1 = tkinter.Entry(root, width=15)  # b(21)
    text_input7_1.grid(row=7, column=1)

    text_input8_1 = tkinter.Entry(root, width=15)  # x[1](0)
    text_input8_1.grid(row=8, column=1)

    text_input9_1 = tkinter.Entry(root, width=15)  # x[2](0)
    text_input9_1.grid(row=9, column=1)

    text_input10_1 = tkinter.Entry(root, width=15)  # Wartość zadana x[1]*
    text_input10_1.grid(row=10, column=1)

    text_input11_1 = tkinter.Entry(root, width=15)  # Wartość zadana x[2]*
    text_input11_1.grid(row=11, column=1)

    text_input12_1 = tkinter.Entry(root, width=15)  # zliczba iteracji1
    text_input12_1.grid(row=12, column=1)

    text_input13_1 = tkinter.Entry(root, width=15)  # gamma
    text_input13_1.grid(row=13, column=1)

    text_input14_1 = tkinter.Entry(root, width=15)  # w(0)1
    text_input14_1.grid(row=14, column=1)

    text_input15_1 = tkinter.Entry(root, width=15)  # w(0)2
    text_input15_1.grid(row=15, column=1)

    text_input16_1 = tkinter.Entry(root, width=15)  # w(1)
    text_input16_1.grid(row=16, column=1)

    text_input17_1 = tkinter.Entry(root, width=15)  # w(2)
    text_input17_1.grid(row=17, column=1)

    text_input18_1 = tkinter.Entry(root, width=15)  # w(3)
    text_input18_1.grid(row=18, column=1)

    text_input19_1 = tkinter.Entry(root, width=15)  # w(4)
    text_input19_1.grid(row=19, column=1)

    text_input20_1 = tkinter.Entry(root, width=15)  # w(5)
    text_input20_1.grid(row=20, column=1)

    text_input21_1 = tkinter.Entry(root, width=15)  # w(6)
    text_input21_1.grid(row=21, column=1)


def historical_data_yes(coun, lis_train_data, lis_iteracji, lis_bledow_error, dane_prun, lis_iter_his,war_zad_iter):
    label_title1_0.destroy()
    button2_0.destroy()
    button2_1.destroy()

    def show_input_parameters2(coun, lis_iter_his):
        if coun == 1:
            weight01 = random()
            weight02 = random()
            weight1 = random()
            weight2 = random()
            weight3 = random()
            weight4 = random()
            weight5 = random()
            weight6 = random()

            w01h, w02h, w1h, w2h, w3h, w4h, w5h, w6h, x_01h, x_02h, lis_error, lis_iter_his, u_nh, x_1np1h, x_2np1h = model_learning(
                data_historicall_symulator, lis_bledow_error, lis_iter_his, weight01, weight02, weight1, weight2,
                weight3, weight4, weight5, weight6, 0.1)

            axs3.plot(lis_iter_his, lis_bledow_error)
            axs3.set_title(
                'Wartość błędu e w poszczególnych iteracjach \n e = 0.5*(wzorcowe wyjście - wyjście sieci)**2',
                fontsize=6)
            axs3.set_xlabel('liczba iteracji', fontsize=6)
            axs3.set_ylabel('wartość błędu', fontsize=6)
            canvas = FigureCanvasTkAgg(fig3, master=root)
            canvas.draw()
            canvas.get_tk_widget().place(x=520, y=510)
            fig3.tight_layout()


        else:
            label_title23_0.destroy()
            button24_0.destroy()
            button24_1.destroy()

        def download_parameters(war_zad_iter):

            x_1star = text_input10_1.get()  # Wartość zadana x[1]*
            x_2star = text_input11_1.get()  # Wartość zadana x[2]*
            iterations = text_input12_1.get()  # liczba iteracji
            gamma = text_input13_1.get()  # gamma

            x_1star = float(x_1star)  # Wartość zadana x[1]*
            x_2star = float(x_2star)  # Wartość zadana x[2]*
            iterations = float(iterations)  # liczba iteracji
            gamma = float(gamma)  # gamma


            label_title1_2 = tkinter.Label(root,
                                           text=f"Parametr uczenia sieci: {gamma}",
                                           bg="white")
            label_title1_2.grid(row=0, column=3, columnspan=2)

            label_title2_3 = tkinter.Label(root,
                                           text=f"wartość zadana x[1]*: {x_1star}, x[2]*: {x_2star} liczba iteracji: {iterations}",
                                           bg="white")
            label_title2_3.grid(row=1, column=3, columnspan=2)
            war_zad_iter.append([(x_1star, x_2star), iterations])
            label_title3_3 = tkinter.Label(root,
                                           text=f"wartości zadane (x[1]*, x[2]*), liczba iteracji:{war_zad_iter}",
                                           bg="white")
            label_title3_3.grid(row=2, column=3, columnspan=2)

            a_11, a_12, a_21, a_22, b_11, b_21 = data_historicall_symulator_parametry
            if coun != 1:

                x_01 = dane_prun[-1][11]
                x_02 = dane_prun[-1][12]
                w01, w02, w1, w2, w3, w4, w5, w6 = dane_prun[-1][3:11]
                hist_data = [True, []]
            else:
                w01, w02, w1, w2, w3, w4, w5, w6, x_01, x_02 = w01h, w02h, w1h, w2h, w3h, w4h, w5h, w6h, x_1np1h, x_2np1h
                hist_data = [True, [u_nh, x_1np1h, x_2np1h], [w01, w02, w1, w2, w3, w4, w5, w6]]
            list_param = [a_11, a_12, a_21, a_22, b_11, b_21, x_01, x_02, x_1star, x_2star, iterations, gamma, w01, w02,
                          w1,
                          w2, w3, w4, w5, w6]

            run(lis_train_data, lis_iteracji, lis_bledow_error, coun, list_param, dane_prun, lis_iter_his, hist_data)

        global button22_0
        button22_0 = tkinter.Button(root, text="Uruchom", bg="white", width=15, command=lambda: download_parameters(war_zad_iter))
        button22_0.grid(row=22, column=0)

    show_input_parameters2(coun, lis_iter_his)


def historical_data_not(coun, lis_train_data, lis_iteracji, lis_bledow_error, dane_prun, lis_iter_his,war_zad_iter):
    label_title1_0.destroy()
    button2_0.destroy()
    button2_1.destroy()

    if coun != 1:
        text_input8_1.config(state="disabled")
        text_input9_1.config(state="disabled")
        text_input14_1.config(state="disabled")
        text_input15_1.config(state="disabled")
        text_input16_1.config(state="disabled")
        text_input17_1.config(state="disabled")
        text_input18_1.config(state="disabled")
        text_input19_1.config(state="disabled")
        text_input20_1.config(state="disabled")
        text_input21_1.config(state="disabled")


    def show_input_parameters(coun):

        if coun != 1:
            label_title23_0.destroy()
            button24_0.destroy()
            button24_1.destroy()

        def download_parameters():
            a_11 = text_input2_1.get()  # a(11)
            a_12 = text_input3_1.get()  # a(12)
            a_21 = text_input4_1.get()  # a(21)
            a_22 = text_input5_1.get()  # a(22)
            b_11 = text_input6_1.get()  # b(11)
            b_21 = text_input7_1.get()  # b(21)
            x_01 = text_input8_1.get()  # x[1](0)
            x_02 = text_input9_1.get()  # x[2](0)
            x_1star = text_input10_1.get()  # Wartość zadana x[1]*
            x_2star = text_input11_1.get()  # Wartość zadana x[2]*
            iterations = text_input12_1.get()  # liczba iteracji
            gamma = text_input13_1.get()  # gamma
            w01 = text_input14_1.get()  # w(01)
            w02 = text_input15_1.get()  # w(02)
            w1 = text_input16_1.get()  # w(1)
            w2 = text_input17_1.get()  # w(2)
            w3 = text_input18_1.get()  # w(3)
            w4 = text_input19_1.get()  # w(4)
            w5 = text_input20_1.get()  # w(5)
            w6 = text_input21_1.get()  # w(6)

            a_11 = float(a_11)  # a(11)
            a_12 = float(a_12)  # a(12)
            a_21 = float(a_21)  # a(21)
            a_22 = float(a_22)  # a(22)
            b_11 = float(b_11)  # b(11)
            b_21 = float(b_21)  # b(21)
            x_01 = float(x_01)  # x[1](0)
            x_02 = float(x_02)  # x[2](0)
            x_1star = float(x_1star)  # Wartość zadana x[1]*
            x_2star = float(x_2star)  # Wartość zadana x[2]*
            iterations = float(iterations)  # liczba iteracji
            gamma = float(gamma)  # gamma
            w01 = float(w01)  # w(01)
            w02 = float(w02)  # w(02)
            w1 = float(w1)  # w(1)
            w2 = float(w2)  # w(2)
            w3 = float(w3)  # w(3)
            w4 = float(w4)  # w(4)
            w5 = float(w5)  # w(5)
            w6 = float(w6)  # w(6)

            label_title1_2 = tkinter.Label(root,
                                           text=f"Parametry obiektu: a(11): {a_11}, a(12): {a_12}, a(21): {a_21}, a(22): {a_22}, b(11): {b_11}, b(21): {b_21} x[1](0): {x_01}, x[2](0): {x_02}, gamma: {gamma},",
                                           bg="white")
            label_title1_2.grid(row=0, column=3, columnspan=2)

            label_title2_3 = tkinter.Label(root,
                                           text=f"wartość zadana x[1]*: {x_1star}, x[2]*: {x_2star} liczba iteracji: {iterations}",
                                           bg="white")
            label_title2_3.grid(row=1, column=3, columnspan=2)
            war_zad_iter.append([(x_1star, x_2star), iterations])
            label_title3_3 = tkinter.Label(root,
                                           text=f"wartości zadane (x[1]*, x[2]*:) liczba iteracji: {war_zad_iter}",
                                           bg="white")
            label_title3_3.grid(row=2, column=3, columnspan=2)

            list_param = [a_11, a_12, a_21, a_22, b_11, b_21, x_01, x_02, x_1star, x_2star, iterations, gamma, w01, w02,
                          w1,
                          w2, w3, w4, w5, w6]
            hist_data = [False, []]
            run(lis_train_data, lis_iteracji, lis_bledow_error, coun, list_param, dane_prun, lis_iter_his, hist_data)

        global button22_0  # 18
        button22_0 = tkinter.Button(root, text="Uruchom", bg="white", width=15, command=lambda: download_parameters())
        button22_0.grid(row=22, column=0)

    show_input_parameters(coun)


counter = 1
lista_train_data = []
lista_iteracji = []
lista_bledow_error = []
data_to_run = []
lis_iter_history = []



def no_cycle(lis_train_data, lis_iteracji, lis_bledow_error, w01, w02, w1, w2, w3, w4, w5, w6, lis_iter_his):

    label_title23_0.destroy()
    button24_0.destroy()
    button24_1.destroy()
    button22_0.destroy()

    def show_end():


        file_2 = open('wyniki_symulacji.json', "r")
        data2 = json.load(file_2)
        file_2.close()

        data2["wyniki"] = lis_train_data

        g = open("wyniki_symulacji.json", "w")
        json.dump(data2, g)
        g.close()

        iterations_for_plot = lis_iteracji
        data_x_1_n = []
        data_x_2_n = []
        data_u_n = []
        for i in lis_train_data:
            data_x_1_n.append(i[0])
            data_x_2_n.append(i[1])
            data_u_n.append(i[2])

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(iterations_for_plot, data_x_1_n, label='x[1](n+1)')
        plt.plot(iterations_for_plot, data_x_2_n, label='x[2](n+1)')
        plt.grid(True)
        plt.xlabel("Liczba iteracji")
        plt.ylabel("Wartość stanu obiektu")
        plt.title(f"Wartość stanu obiektu w poszczególnych iteracjach")
        plt.legend()
        plt.savefig("wykres1.jpg", dpi=164)

        plt.figure(figsize=(10, 6), dpi=70)
        plt.plot(iterations_for_plot, data_u_n)
        plt.grid(True)
        plt.xlabel("Liczba iteracji")
        plt.ylabel("Wartość wielkości sterującej")
        plt.title(f"Wartość wielkośći sterującej w poszczególnych iteracjach")
        plt.savefig("wykres2.jpg", dpi=164)

        plt.figure(figsize=(10, 6), dpi=40)
        plt.plot(lis_iter_his, lis_bledow_error)
        plt.grid(True)
        plt.xlabel("Liczba iteracji")
        plt.ylabel("Wartość błedu")
        plt.title(f"Wartość błedu w poszczególnych iteracjach")
        plt.savefig("wykres3.jpg", dpi=164)

    show_end()


def run(lis_train_data, lis_iteracji, lis_bledow_error, coun, list_param, dane_prun, lis_iter_his, hist_data):
    print('!!! lista parametrów:', list_param)


    if coun == 1:
        button22_0.destroy()
        order = True
        a_11, a_12, a_21, a_22, b_11, b_21, x_01, x_02, x_1star, x_2star, iterations, gamma, w01, w02, w1, w2, w3, w4, w5, w6 = list_param
        liczba_iteracji_gui = iterations

    else:
        button22_0.destroy()
        [a_11, a_12, a_21, a_22, b_11, b_21, x_01, x_02, x_1star, x_2star, iterations, gamma, w01, w02, w1, w2, w3, w4,
         w5, w6] = list_param
        [lis_train_data, lis_iteracji, lis_bledow_error, w01, w02, w1, w2, w3, w4, w5, w6, x_01, x_02, lis_iter_his] = \
        dane_prun[-1]
        liczba_iteracji_gui = lis_iteracji[-1] + list_param[10]
        order = False

    train_data2, lista_iteracji2, lista_bledow_error2, w01, w02, w1, w2, w3, w4, w5, w6, x_01, x_02, lis_iter_his = run_mpc(
        w01, w02,
        w1, w2,
        w3, w4,
        w5, w6,
        x_01,
        x_02,
        liczba_iteracji_gui,
        lis_train_data,
        lis_iteracji,
        lis_bledow_error,
        order,
        a_11,
        a_12,
        a_21,
        a_22,
        b_11,
        b_21,
        gamma,
        x_1star,
        x_2star,
        hist_data, lis_iter_his, coun)

    dane_prun.append(
        [train_data2, lista_iteracji2, lista_bledow_error2, w01, w02, w1, w2, w3, w4, w5, w6, x_01, x_02, lis_iter_his])
    data_x_1_n = []
    data_x_2_n = []
    data_u_n = []
    for i in train_data2:
        data_x_1_n.append(i[0])
        data_x_2_n.append(i[1])
        data_u_n.append(i[2])
    plt.close('all')
    fig1, axs1 = plt.subplots(1, figsize=(3.5, 2.5))
    fig2, axs2 = plt.subplots(1, figsize=(3.5, 2.5))
    fig3, axs3 = plt.subplots(1, figsize=(3.5, 2.5))

    axs1.plot(lista_iteracji2, data_u_n)
    axs1.set_title('Wartość decyzji sterującej w poszczególnych iteracjach', fontsize=6)
    axs1.set_xlabel('liczba iteracji', fontsize=6)
    axs1.set_ylabel('wartość u(n)', fontsize=6)
    canvas = FigureCanvasTkAgg(fig1, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=520, y=70)
    fig1.tight_layout()
    axs2.plot(lista_iteracji2, data_x_1_n, label='x[1](n+1)')
    axs2.plot(lista_iteracji2, data_x_2_n, label='x[2](n+1)')
    axs2.set_title('Wartość stanu obiektu w poszczególnych iteracjach', fontsize=6)
    axs2.set_xlabel('liczba iteracji', fontsize=6)
    axs2.set_ylabel('wartość x(n+1)', fontsize=6)
    axs2.legend()
    canvas = FigureCanvasTkAgg(fig2, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=1065, y=70)
    fig2.tight_layout()
    toolbarFrame = Frame(master=root)
    toolbarFrame.place(x=1050, y=0)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    axs3.plot(lis_iter_his, lista_bledow_error2)
    axs3.set_title('Wartość błędu e w poszczególnych iteracjach \n e = 0.5*(wzorcowe wyjście - wyjście sieci)**2',
                   fontsize=6)
    axs3.set_xlabel('liczba iteracji', fontsize=6)
    axs3.set_ylabel('wartość błędu', fontsize=6)
    canvas = FigureCanvasTkAgg(fig3, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=520, y=510)
    fig3.tight_layout()

    def cycle_return(coun):
        button22_0.destroy()
        global label_title23_0
        global button24_0
        global button24_1
        label_title23_0 = tkinter.Label(root, text="Czy chcesz dalej przeprowadzać obliczenia?", bg="white")  # 14
        label_title23_0.grid(row=23, column=0, columnspan=2)  # 14
        coun += 1

        def y_c(coun, train_data2, lista_iteracji2, lista_bledow_error2, historical_data, dane_prun, lis_iter_his):
            if hist_data[0] == True:
                historical_data_yes(coun, train_data2, lista_iteracji2, lista_bledow_error2, dane_prun, lis_iter_his, war_zad_iter)
            else:
                historical_data_not(coun, train_data2, lista_iteracji2, lista_bledow_error2, dane_prun, lis_iter_his, war_zad_iter)


        button24_0 = tkinter.Button(root, text="TAK", bg="white", width=15,
                                    command=lambda: y_c(coun, train_data2, lista_iteracji2, lista_bledow_error2,
                                                        hist_data, dane_prun, lis_iter_his))
        button24_0.grid(row=24, column=0)


        button24_1 = tkinter.Button(root, text="NIE", bg="white", width=15,
                                    command=lambda: no_cycle(lis_train_data, lis_iteracji, lis_bledow_error, w01, w02,
                                                             w1, w2, w3, w4, w5, w6, lis_iter_his))
        button24_1.grid(row=24, column=1)

    cycle_return(coun)

    return train_data2, lista_iteracji2, lista_bledow_error2


root.mainloop()