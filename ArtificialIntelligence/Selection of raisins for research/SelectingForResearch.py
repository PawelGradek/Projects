#CTM
import json

file_1 = open('besni_set.json', "r")
data = json.load(file_1)
file_1.close()

file_2 = open('unnormed.json', "r")
data2 = json.load(file_2)
file_2.close()

unnormed_set = data2["unnormed.json"]
besni_set1 = data["besni_set"]

besni_set = []
for i in besni_set1:
    for j in unnormed_set:
        if i[8] == j[8]:
            besni_set.append(j)
# print(besni_set)

Value = []
Weight = []
capacity = 800000
for i in besni_set:
    Value.append(i[6])
    Weight.append(i[0])


def begin_solution_fun(set, Value, Weight, capacity):
    item = {}
    capacity = capacity
    for i in range(len(set)):
        item[i] = Value[i]/Weight[i], Value[i], Weight[i], i
    item = sorted(item.values(), reverse=True)
    value = 0
    weight = 0
    solution = []
    for i in range(len(Value)):
        if item[i][2] <= capacity:
            capacity -= item[i][2]
            weight += item[i][2]
            solution.append(item[i][3])
            value += item[i][1]
    begin_solution = [0 for i in range(len(Value))]
    for i in solution:
        begin_solution[i] = 1

    return begin_solution, weight

def count_value(begin_solution, item, capacity):
    max_value = 0
    for i in range(len(item)):
        if begin_solution[i] == 1 and capacity >= 0:
            max_value += item[i][0]
            capacity -= item[i][1]
    if capacity < 0:
        max_value = -1
    return max_value

def raisin_id(best_solution, item):
    raisin_id = []
    for i in range(len(item)):
        if best_solution[i] == 1:
            raisin_id.append(item[i][2])
    return raisin_id


begin_solution, current_weight = begin_solution_fun(besni_set, Value, Weight, capacity)


def tabu_search(begin_solution, value, weight, capacity, set):
    item = {}
    for i in range(len(set)):
        item[i] = value[i], weight[i], set[i][8]
    tabu = []
    iteration = 0
    iteration_bs_sol = 0
    best_solution = begin_solution[:]
    capacity2 = capacity
    interim_list = best_solution[:]
    while iteration - iteration_bs_sol <= 200:
        iteration += 1
        save_sol = interim_list[:]
        value_capacity = -1
        iter = -1
        for i in range(len(item)):
            if interim_list[i] == 1:
                interim_list[i] = 0
            else:
                interim_list[i] = 1

            solution_fract = interim_list[:]
            if count_value(solution_fract, item, capacity2) > value_capacity and not (i in tabu):
                save_sol = solution_fract[:]
                value_capacity = count_value(save_sol, item, capacity)
                iter = i
                if count_value(solution_fract, item, capacity2) > count_value(best_solution, item, capacity2):
                    best_solution = solution_fract[:]
                    iteration_bs_sol = iteration

            if interim_list[i] == 1:
                interim_list[i] = 0
            else:
                interim_list[i] = 1
        if iter != -1:
            if not (iter in tabu):
                if len(tabu) == 20:
                    del tabu[0]
                tabu.append(iter)
            interim_list = save_sol[:]

    print('The total sum of the raisin circumference to be tested', count_value(best_solution, item, capacity))
    print("List of raisin id's that will be taken for research", raisin_id(best_solution,item))

    return raisin_id(best_solution, item)

x = tabu_search(begin_solution, Value, Weight, capacity, besni_set)

end_value_area = 0
for i in besni_set:
    for j in x:
        if i[8] == j:
            end_value_area += i[0]

print('Total surface area of raisins to be tested', end_value_area)
