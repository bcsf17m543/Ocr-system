import numpy as np
from matplotlib import pyplot as plt


def ReadnConvert():
    data = np.genfromtxt('trainX.txt', dtype=np.int32)
    return data


def Readfilex():
    with open('trainX.txt') as file:
        array2d = [[int(digit) for digit in line.split()]
                   for line in file]
    file.close()
    return array2d


def Convert(x_train):
    num_lines = sum(1 for line in open('trainX.txt'))
    x_arr = np.array([num_lines, 256])
    count = 0
    for i in x_train:
        x_arr[count] = np.array(x_train[count])
        count = count+1
    return x_arr


def Two_Train(x_train):
    count_2 = 0
    file1 = open("trainY.txt", "r")
    lines = file1.read().split()
    two_cat = [f for f in lines if f == '2']
    file1.close()
    count_2 = len(two_cat)
    two_train = x_train[:count_2, :]  # trainig data for 2
    two_one_Prob = (two_train.sum(axis=0)+1)/(two_train.shape[0]+2)
    return two_one_Prob, count_2


def Four_Train(count, x_train):
    num_lines = sum(1 for line in open('trainY.txt'))
    count_4 = num_lines-count
    four_train = x_train[count:, :]  # trainig data for 2
    four_one_Prob = (four_train.sum(axis=0)+1)/(four_train.shape[0]+2)
    four_zero_Prob = 1-four_one_Prob
    return four_one_Prob, four_zero_Prob, count_4


def Class_Prob(count2, count4, total):
    p_two = count2/total
    p_four = count4/total
    return p_two, p_four


def Testing(two_one_Prob, two_zero_Prob, four_one_Prob, four_zero_Prob, p2, p4):
    x_test = np.genfromtxt('testX.txt', dtype=np.int32)  # testx
    y_test = np.genfromtxt('testY.txt', dtype=np.int32)
    ans_2 = np.empty(x_test.shape[0], dtype=float)
    ans_4 = np.empty(x_test.shape[0], dtype=float)
    # itr = 0
    for i in range(x_test.shape[0]):
        val1, val2 = 1, 1
        for j in range(x_test.shape[1]):
            if x_test[i][j] == 1:
                # ans_2[i] = ans_2[i] * two_one_Prob[j]
                val1 = val1*two_one_Prob[j]
                # ans_4[i] = ans_4[i] * four_one_Prob[j]
                val2 = val2*four_one_Prob[j]
            if x_test[i][j] == 0:
                # ans_2[i] = ans_2[i] * two_zero_Prob[j]
                val1 = val1 * two_zero_Prob[j]
                # ans_4[i] = ans_4[i] * four_zero_Prob[j]
                val2 = val2 * four_zero_Prob[j]
        ans_2[i] = val1*p2
        ans_4[i] = val2*p4
    ans_2 = ans_2 * p2
    ans_4 = ans_4 * p4
    ans_gen = np.empty(x_test.shape[0], dtype=int)
    for i in range(ans_gen.shape[0]):
        if(ans_2[i] < ans_4[i]):
            ans_gen[i] = 4
        else:
            ans_gen[i] = 2
    return ans_gen


def Calculate_Accuracy(ans_gen):
    data = np.genfromtxt('testY.txt', dtype=np.int32)
    TP = sum(ans_gen == data)
    TN = sum(ans_gen != data)
    FN = 1-TP
    FP = 1-TN
    acc = TP+TN/(TN+TP+FN+FP)
    print(acc)

    ################  main ##################
x_train = ReadnConvert()
# temp = np.reshape(x_train[2], (16, 16), order='F')
# plt.imshow(temp)
two_one_Prob, count = Two_Train(x_train)
two_zero_Prob = 1-two_one_Prob
four_one_Prob, four_zero_Prob, count4 = Four_Train(count, x_train)
total = sum(1 for line in open('trainY.txt'))
p2, p4 = Class_Prob(count, count4, total)
ans_gen = Testing(two_one_Prob, two_zero_Prob,
                  four_one_Prob, four_zero_Prob, p2, p4)
Calculate_Accuracy(ans_gen)
