import numpy as np
from io import StringIO
import math

def my_stddev(a):
    sum = 0
    for value in a:
        sum += value
    average = sum/len(a)
    average2 = average*average
    sum2 = 0
    for value in a:
        sum2 += (value*value) - average2
    return math.sqrt(sum2/len(a))

def cost(a, t, mean, stdev):
    sum = 0
    for i in range(0, len(a)-t-1):
        sum += (a[i] - mean)*(a[t+i] - mean)
    return sum/stdev/stdev/(len(a)-t)

def my_actime(a):
    mean = np.mean(a)
    stdev = np.std(a)
    sum = 0
    t = 0
    c = 0
    while True:
        t += 1
        c = cost(a, t, mean, stdev)
        if c <= 0:
            break
        sum += c
    return 1 + 2*sum

def my_stderr(a):
    neff = len(a)/my_actime(a)
    return np.std(a)/np.sqrt(neff)

def my_err(a):
    return np.std(a)/np.sqrt(len(a))

#data = open("C:/Users/Justin Baker/Downloads/data2.11 (3).txt", "r")
#lines = data.readlines()

#list = []
#listA = [1.12, 1.52, 1.33, 1.09, 1.20, 1.26]
#listB = [1.44, 1.34, 1.19, 1.13, 1.56, 1.45]
#for line in lines:
#    newline = line.split(",")[0]
#    list.append(newline)

#i = 0
#for line in lines:
#    if (i >= 1000):
#            i += 1
#            continue
#    newline = line.split(",")[0]
#    list.append(newline)
#    i += 1
#print(len(list))

#ar = np.asarray(listA, float)
#print(ar)
#print(lines)
#data = np.genfromtxt("C:/Users/Justin Baker/Downloads/data1.5.txt", delimiter = ",")
#data = np.array(data)
#print(ar)
#mean = np.mean(ar)
#stddev = np.std(ar)
#stderr = my_err(ar)
#print(mean)
#print(stddev)
#print(stderr)