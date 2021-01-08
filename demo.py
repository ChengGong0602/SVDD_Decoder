import numpy as np
import random as rd
import pylab
import matplotlib
import matplotlib.pyplot as plt
from gurobipy import *
import math
import time

def maximize_L(training_set):
    m = Model("Dual Lagrangian")
    vars = ['']*int(round(element_num*training_rate))
    for i in range(int(round(element_num*training_rate))):
        vars[i] = m.addVar(vtype=GRB.CONTINUOUS, name="x"+str(i))
    m.update()

    ## Objective function 
    obj_1 = sum(vars[i]*inner_product(training_set[i], training_set[i]) for i in range(int(round(element_num*training_rate))))
    obj_2 = 0
    for i in range(int(round(element_num*training_rate))):
        for j in range(int(round(element_num*training_rate))):
            obj_2 += vars[i]*vars[j]*inner_product(training_set[i], training_set[j])
    obj = obj_1 - obj_2

    m.setObjective(obj, GRB.MAXIMIZE)
    for i in range(int(round(element_num*training_rate))):
        m.addConstr(vars[i] <= C, "c"+str(i))
    m.optimize()

    ## Rounding up
    solution = []
    for v in m.getVars():
        solution.append(round(v.x, int(math.floor(abs(np.log10(C))+2))))
    return solution

def select_sv_on(alpha_dict):
    sv_on_point_index = []
    for i in range(int(round(element_num*training_rate))):
        if alpha_dict[i] > 0:
            if alpha_dict[i] < C:
                sv_on_point_index.append(i)
    print ("number sv_on: "+str(len(sv_on_point_index)))
    return sv_on_point_index

def select_sv_outsider(alpha_dict):
    sv_outsider_point_index = []
    for i in range(int(round(element_num*training_rate))):
        if alpha_dict[i] == C :
            sv_outsider_point_index.append(i)
    print ( "number sv_outsider: "+str(len(sv_outsider_point_index)))
    return sv_outsider_point_index

def select_nsv(sv_on, sv_outsider):
    index = [i for i in range(int(round(element_num*training_rate)))]
    sv = sv_on + sv_outsider
    for i in sv:
        index.remove(i)
    print ("number nsv: "+str(len(index)))
    return index

def test_SVDD(test_set, data_class, middlepoint, R, point_dict, alpha_dict):
    FRR2, FAR2 = 0, 0
    for i in range(len(test_set)):
        if data_class[i] == 1:
            FRR2 +=1
        else:
            FAR2 +=1
    i, FRR1, FAR1 = 0, 0, 0
    for test_data in test_set:
        if calculate_radius(test_data, point_dict, alpha_dict, middlepoint) > R :
            print("Data #" + str(i+1) + " : No")
            if data_class[i] == 1:
                FRR1 += 1
        else:
            print( "Data #" + str(i+1) + " : Yes")
            if data_class[i] == 2:
                FAR1 += 1
        i+=1
    print ("\nTotal set : " + str(len(test_set)))
    print ("FRR : " + str(FRR1/float(FRR2)))
    print ("FAR : " + str(FAR1/float(FAR2)))
    print ("Radius R : " + str(R))
    return FRR1/float(FRR2), FAR1/float(FAR2)
def plot(sv_on, sv_outsider, nsv, point_dict, middlepoint, R, alpha_dict):

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    delta = 0.5
    x_min, x_max, y_min, y_max = -5, 5, -5, 5
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    Z_list = []
    for i in range(int((x_max-x_min)/delta)):
        Z_list_row = []
        for j in range(int((x_max-x_min)/delta)):
            point = [X[i][j], Y[i][j]]
            Z_list_row.append(calculate_radius(point, point_dict, alpha_dict, middlepoint))
            print(j)
        Z_list.append(Z_list_row)
        print (i)
    Z = np.array(Z_list)
    plt.figure()
    CS = plt.contour(X, Y, Z, colors='black', linestyles = 'dashed')
    CSR = plt.contour(X, Y, Z, levels = [R], colors='red', linestyles = 'solid')
    plt.clabel(CS, inline=1, fontsize=7)
    plt.clabel(CSR, inline=1, fontsize=7)

    for k in sv_on:
        plt.plot(point_dict[k][0], point_dict[k][1], 'ro')
    plt.show()
train_set = np.randdom.rand(10,20)
maximize_L(train_set)