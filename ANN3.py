from tkinter import *
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def dataclassification():
    data = pd.read_csv("penguins.csv")
    # ---------------------------------------------
    gender = data['gender'].replace(['female', 'male', np.nan], [0, 1, 0], inplace=True)  # encoding the gender
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
    species = ['Adelie', 'Gentoo', 'Chinstrap']

    # Features Scaling
    standard = StandardScaler()
    standard.fit(data.iloc[:, 1:])
    data.iloc[:, 1:] = standard.transform(data.iloc[:, 1:])

    x_train1 = data.iloc[:30, 1:]
    x_train2 = data.iloc[50:80, 1:]
    x_train3 = data.iloc[100:130, 1:]
    xtrain = [x_train1, x_train2, x_train3]
    x_train = pd.concat(xtrain).reset_index(drop=True)

    y_train1 = data.iloc[:30, 0]
    y_train2 = data.iloc[50:80, 0]
    y_train3 = data.iloc[100:130, 0]
    ytrain = [y_train1, y_train2, y_train3]
    y_train = pd.concat(ytrain).reset_index(drop=True)

    x_test1 = data.iloc[30:50, 1:]
    x_test2 = data.iloc[80:100, 1:]
    x_test3 = data.iloc[130:,1:]
    xtest = [x_test1, x_test2, x_test3]
    x_test = pd.concat(xtest).reset_index(drop=True)

    y_test1 = data.iloc[30:50, 0]
    y_test2 = data.iloc[80:100, 0]
    y_test3 = data.iloc[130:, 0]
    ytest = [y_test1, y_test2, y_test3]
    y_test = pd.concat(ytest).reset_index(drop=True)

    y_train = pd.get_dummies(y_train, drop_first=False)
    y_test = pd.get_dummies(y_test, drop_first=False)
    return x_train, y_train, x_test, y_test
    #Adelie 1 0 0
    #Gentoo 0 0 1
 #Chinstrap 0 1 0
root = Tk()
root.title("Back Propagation")
root.geometry('700x200')
root['background'] = '#FFFFE4'
funOptions = ["Sigmoid", "Tanget"]
# datatype of menu text
clicked = StringVar()
# initial menu text
clicked.set("Activation Functions")
drop = OptionMenu(root, clicked, *funOptions)
drop.place(x=410, y=130)
neurons = tk.Label(root, text="Neurons").place(x=350, y=50)
neuron = Entry(root)
neuron.place(x=410, y=50)
neuron.focus_set()

layers = tk.Label(root, text="Layers").place(x=350, y=100)
layer = Entry(root)
layer.place(x=410, y=100)
layer.focus_set()
Enterlr = tk.Label(root, text="Learning Rate").place(x=20, y=50)
etatv = Entry(root)
etatv.place(x=100, y=50)
etatv.focus_set()
Enterlr = tk.Label(root, text="Epochs").place(x=20, y=100)
epochstv = Entry(root)
epochstv.place(x=100, y=100)
epochstv.focus_set()
var = IntVar()
c1 = tk.Checkbutton(root, text='Bias', variable=var, onvalue=1, offvalue=0).place(x=20, y=140)
bias = 0
etanum = 0
epochsnum = 0
layers = 0
numofneuorans=[]
functions = ''
def submit():
        global bias
        global etanum
        global epochsnum
        global layers
        global numofneuorans
        global functions
        etanum = float(etatv.get())
        epochsnum = int(epochstv.get())
        layers = int(layer.get())
        numofneuorans= [int(i) for i in neuron.get().split(',')]
        functions = clicked.get()
        if var.get() == 1:
            bias = True
        else:
            bias = False
subButton = tk.Button(root, text="Submit", command=submit, bg='#FF99CC', fg='white').place(x=250, y=150)
root.mainloop()
xtrain , ytrain , xtest , ytest = dataclassification()

def sigmoid(net):
    return 1 / (1 + np.exp(-net))
def hyperbolicTangentSigmoid(net):
    return np.tanh(net)
def feedforward(X, outputneurons, isbiased, actfun, numoflayers, numofneuorans, firstiteration=True, W=None):
    totallayernum = numoflayers + 1
    W = []
    fx = []
    # for each hidden layer
    p =[]
    for l in range(totallayernum):  # weight matrix = [n[l],n[l-1]] 3x6    0 1 2
        if l != numoflayers:
            if l == 0:
                if firstiteration:
                    p = np.random.uniform(low=-1, high=1, size=[numofneuorans[l], X.shape[0]])
                    currentn = numofneuorans[l]
            else:
                if firstiteration:
                    p = np.random.uniform(low=-1, high=1, size=[numofneuorans[l], numofneuorans[l-1]])
                    currentn = numofneuorans[l]
        else:  # l==numoflayers
            if firstiteration:
                p = np.random.uniform(low=-1, high=1, size=[outputneurons, numofneuorans[l - 1]])
                currentn = outputneurons
        if isbiased == False:
            p[:, 0] = 0
        net = np.dot(p, X)
        if actfun == 'Sigmoid':
             fx.append(sigmoid(net))
        else:
            fx.append(hyperbolicTangentSigmoid(net))
        if l != numoflayers:
            fx[l][0] = 1
        # print("l",l)
        # print("w",p.shape)
        # print("x",X.shape)
        fx[l] = np.reshape(fx[l], (currentn, 1))
        X = fx[l]
        W.append(p)
    return W, fx

def training(xtrain, ytrain, isbiased, eta, epochs, numoflayers, numofneuorans, actfun):
    ytrain = np.expand_dims(ytrain, axis=1)
    xtrain = np.c_[np.ones((xtrain.shape[0], 1)), xtrain]   #bzwd col bias b 0 fl xtrain bnfs l shakl
    label = []
    weight = np.empty(xtrain.shape[0], dtype=object)
    for epoch in range(epochs):
        for i in range(xtrain.shape[0]):
            input = np.expand_dims(xtrain[i], axis=1)
            label = [item for i in ytrain[i] for item in i]     #1 0 0
            outputneuronshape = len(label)
            weight[i],fx= feedforward(input,outputneuronshape , isbiased, actfun, numoflayers, numofneuorans)
            while True:
                Error = backstep(label,numoflayers,weight[i],fx)
                if all(Error[numoflayers] < 0.1):
                    break
                weight[i] = upadte(weight[i], eta, Error, input,fx, numoflayers)
                weight[i], fx = feedforward(input, outputneuronshape, isbiased, actfun, numoflayers, numofneuorans, True, weight[i])
    # print(weight)
    return weight

def backstep(ytrain,numoflayers,w,fx):
    lis = [item for i in fx[numoflayers] for item in i]
    e = [None] * (numoflayers + 1)  #matrix b row wa7d w cols b3dd no of layers +1
    outputerror = np.subtract(ytrain, lis)
    delta = np.dot(np.transpose(lis[numoflayers]), (1 - lis[numoflayers]))
    # print(delta)
    # print(outputerror)
    e[numoflayers] = np.dot(outputerror, delta)
    # print(e[numoflayers])
    for i in reversed(range(numoflayers)):
        e[i] = np.dot(e[i+1], w[i+1])*((lis[i])*(1 - lis[i]))
    return e

def upadte(w,eta,e,x,fx,numoflayers):
    for l in range(numoflayers+1):
        elis = [item for i in e for item in i]
        fxlis = [item for i in fx[numoflayers] for item in i]
        if l == 0:  # w = w + eta * error * x
           w[l] += eta*np.dot(elis[l], np.transpose(x))   #21    16
        else:
           # rar=np.dot(np.transpose(fxlis), elis[l])
           w[l] = w[l]+eta*np.dot(elis[l], np.transpose(fxlis[l - 1]))
    return w

def testing(xtest,bias,functions,layers,numofneuorans, ytest):
    ytest = np.expand_dims(ytest, axis=1)   #kolhom f column wa7d
    xtest = np.c_[np.ones((xtest.shape[0], 1)), xtest]
    weight = np.empty(xtest.shape[0], dtype=object)
    label = []
    prrdicted = []
    totalpredicted=[]
    for i in range(xtest.shape[0]):
        input = np.expand_dims(xtest[i], axis=1)
        label = [item for i in ytest[i] for item in i]
        outputneuronshape = len(label)
        weight[i], fx = feedforward(input, outputneuronshape, bias, functions, layers, numofneuorans)
        prrdicted.append(fx[-1])
    for m in range(len(prrdicted)):
        ListFX = [item for i in prrdicted[m] for item in i] #0.4 0.2 0.3     1 0 0
        # print(ListFX)
        listOutput = []
        for i in ListFX:
            if (i == max(ListFX)):
                x = 1
                listOutput.append(x)
            else:
                x = 0
                listOutput.append(x)
            totalpredicted.append(listOutput)
    counter=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(ytest.shape[0]):
        label = [item for i in ytest[i] for item in i]
        # print(label)
        # print(totalpredicted[i])
        if i < 20:
            if label == totalpredicted[i]:
                counter += 1
                tp += 1
            else:
                fp += 1
        else:
            if label == totalpredicted[i]:
                counter += 1
                tn += 1
            else:
                fn += 1
    confusionMatrix = np.array([[tp, fp], [fn, tn]])
    print("Confusion Matrix")
    print(confusionMatrix)
    print("Overall accuracy is ", ((tp + tn) / 60) * 100)

trainweight = training(xtrain,ytrain,bias,etanum,epochsnum,layers,numofneuorans,functions)
print("Training Accuracy: ")
testing(xtrain, bias, functions, layers, numofneuorans, ytrain)
print("------------------------------------------------------")
print('Testing Accuracy: ')
testing(xtest,bias,functions, layers,numofneuorans, ytest)
print("------------------------------------------------------")
