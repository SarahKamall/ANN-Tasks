from tkinter import *
import tkinter as tk
import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# reading CSV file
data = read_csv("penguins.csv")
#---------------------------------------------
gender = data['gender'].replace(['female', 'male', np.nan], [0, 1, 0], inplace=True) #encoding the gender
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm','gender', 'body_mass_g']
species = ['Adelie', 'Gentoo', 'Chinstrap']

#shuffle data
shuffleData = data.sample(frac=1)

# Features Scaling
# standard = StandardScaler()
# standard.fit(data.iloc[:, 1:])
# data.iloc[:, 1:] = standard.transform(data.iloc[:, 1:])

#Species
x1 = data.iloc[0:50, :] #Adelie species
x2 = data.iloc[50:100, :] #Gentoo species
x3 = data.iloc[100:151, :] #Chinstrap species

#train samples
train_x1 = data.iloc[0:30, :]
# normalizedx1= preprocessing.minmax_scale(train_x1)
train_x2 = data.iloc[50:80, :]
# normalizedx2= preprocessing.normalize(train_x2)
train_x3 = data.loc[100:130, :]
# normalizedx3= preprocessing.normalize(train_x3)

#test samples
test_x1 = data.iloc[30:50, :]
test_x2 = data.iloc[80:100, :]
test_x3 = data.iloc[130:150, :]

root = Tk() #window
root.title("Feature Selection")
root.geometry('800x800')
root['background'] = '#CCE5FF'

listoffeature = []
listoflabel = []
error=[]

featurelb = Listbox(root, selectmode=MULTIPLE, height=5, width=50)  # create Listbox
labellb = Listbox(root, selectmode=MULTIPLE, height=3, width=50)  # create Listbox

def printfeature():
    feture = featurelb.curselection()
    for i in feture:
        listoffeature.append(featurelb.get(i))

def printlabel():
    label = labellb.curselection()
    for i in label:
        listoflabel.append(labellb.get(i))

def listfeature():
    for x in features:
        featurelb.insert(END, x)
    featurelb.pack(pady=15)  # put listbox on window
listfeature()
featureButton = Button(root, text="Select Feature", command=printfeature, bg='#FF99CC', fg='white').pack()

def listlabel():
    for x in species:
        labellb.insert(END, x)
    labellb.pack(pady=10)  # put listbox on window
listlabel()
labelButton = Button(root,text="Select Label",command=printlabel,bg='#FF99CC', fg='white').pack()
Enterlr = tk.Label(root, text="Learning Rate").place(x=20, y=50)
etatv = Entry(root)
etatv.place(x=100, y=50)
etatv.focus_set()

Enterep = tk.Label(root, text="Epochs").place(x=20, y=100)
epochstv = Entry(root)
epochstv.place(x=100, y=100)
epochstv.focus_set()
Enterth = tk.Label(root, text="threashold").place(x=20, y=200)
threasholdtv = Entry(root)
threasholdtv.place(x=100, y=200)
threasholdtv.focus_set()
var = IntVar()

c1 = tk.Checkbutton(root, text='bias', variable=var, onvalue=1, offvalue=0).place(x=20,y=140)
def submit():
    global bias
    global threashold
    global eta
    global epochs
    eta = float(etatv.get())
    threashold = float(threasholdtv.get())
    epochs = int(epochstv.get())
    if var.get() == 1:
        bias = 1
    else:
        bias = 0
subButton = tk.Button(root,text="Submit",command=submit, bg='#FF99CC', fg='white').place(x=120,y=250)

def visulaization():
    plt.figure('figure1')
    plt.scatter((data.loc[0:49,"bill_length_mm"]),(data.loc[0:49,"bill_depth_mm"]),c="blue")
    plt.scatter((data.loc[50:100,"bill_length_mm"]), (data.loc[50:100,"bill_depth_mm"]),c="pink")
    plt.scatter((data.loc[100:150,"bill_length_mm"]), (data.loc[100:150,"bill_depth_mm"]),c="green")
    plt.xlabel("bill length")
    plt.ylabel("bill debth ")
    plt.show()

    plt.figure('figure2')
    plt.scatter((data.loc[0:49,"bill_length_mm"]), (data.loc[0:49,"flipper_length_mm"]),c="blue")
    plt.scatter((data.loc[50:100,"bill_length_mm"]), (data.loc[50:100,"flipper_length_mm"]),c="pink")
    plt.scatter((data.loc[100:150,"bill_length_mm"]),(data.loc[100:150,"flipper_length_mm"]),c="green")
    plt.xlabel("bill length")
    plt.ylabel("flipper_length")
    plt.show()

    plt.figure('figure3')
    plt.scatter((data.loc[0:49, "bill_length_mm"]), (data.loc[0:49, "gender"]),c="blue")
    plt.scatter((data.loc[50:100, "bill_length_mm"]), (data.loc[50:100, "gender"]),c="pink")
    plt.scatter((data.loc[100:150, "bill_length_mm"]), (data.loc[100:150, "gender"]),c="green")
    plt.xlabel("bill length")
    plt.ylabel("gender")
    plt.show()

    plt.figure('figure4')
    plt.scatter((data.loc[0:49, "bill_length_mm"]), (data.loc[0:49, "body_mass_g"]),c="blue")
    plt.scatter((data.loc[50:100, "bill_length_mm"]), (data.loc[50:100, "body_mass_g"]),c="pink")
    plt.scatter((data.loc[100:150, "bill_length_mm"]), (data.loc[100:150, "body_mass_g"]),c="green")
    plt.xlabel("bill length")
    plt.ylabel("body mass")
    plt.show()

    plt.figure('figure5')
    plt.scatter((data.loc[0:49,"bill_depth_mm"]), (data.loc[0:49,"flipper_length_mm"]),c="blue")
    plt.scatter((data.loc[50:100,"bill_depth_mm"]), (data.loc[50:100,"flipper_length_mm"]),c="pink")
    plt.scatter((data.loc[100:150,"bill_depth_mm"]),(data.loc[100:150,"flipper_length_mm"]),c="green")
    plt.xlabel("bill depth")
    plt.ylabel("flipper_length")
    plt.show()

    plt.figure('figure6')
    plt.scatter((data.loc[0:49, "bill_depth_mm"]), (data.loc[0:49, "gender"]),c="blue")
    plt.scatter((data.loc[50:100, "bill_depth_mm"]), (data.loc[50:100, "gender"]),c="pink")
    plt.scatter((data.loc[100:150, "bill_depth_mm"]), (data.loc[100:150, "gender"]),c="green")
    plt.xlabel("bill depth")
    plt.ylabel("gender")
    plt.show()

    plt.figure('figure7')
    plt.scatter((data.loc[0:49, "body_mass_g"]),(data.loc[0:49, "bill_depth_mm"]),c="blue")
    plt.scatter((data.loc[50:100, "body_mass_g"]),(data.loc[50:100, "bill_depth_mm"]),c="pink")
    plt.scatter((data.loc[100:150, "body_mass_g"]),(data.loc[100:150, "bill_depth_mm"]),c="green")
    plt.xlabel("body mass")
    plt.ylabel("bill depth")
    plt.show()

    plt.figure('figure8')
    plt.scatter((data.loc[0:49,"flipper_length_mm"]),(data.loc[0:49, "gender"]),c="blue")
    plt.scatter((data.loc[50:100,"flipper_length_mm"]),(data.loc[50:100, "gender"]),c="pink")
    plt.scatter((data.loc[100:150,"flipper_length_mm"]),(data.loc[100:150, "gender"]),c="green")
    plt.xlabel("flipper length")
    plt.ylabel("gender")
    plt.show()

    plt.figure('figure9')
    plt.scatter((data.loc[0:49, "flipper_length_mm"]), (data.loc[0:49, "body_mass_g"]),c="blue")
    plt.scatter((data.loc[50:100, "flipper_length_mm"]), (data.loc[50:100, "body_mass_g"]),c="pink")
    plt.scatter((data.loc[100:150, "flipper_length_mm"]), (data.loc[100:150, "body_mass_g"]),c="green")
    plt.xlabel("flipper length")
    plt.ylabel("body mass")
    plt.show()

    plt.figure('figure10')
    plt.scatter((data.loc[0:49, "gender"]), (data.loc[0:49, "body_mass_g"]),c="blue")
    plt.scatter((data.loc[50:100, "gender"]), (data.loc[50:100, "body_mass_g"]),c="pink")
    plt.scatter((data.loc[100:150, "gender"]), (data.loc[100:150, "body_mass_g"]),c="green")
    plt.xlabel("gender")
    plt.ylabel("body mass")
    plt.show()
visualizebutton = tk.Button(root,text="Visualization",command=visulaization, bg='#FF99CC', fg='white').place(x=120,y=300)
root.mainloop()

def signum(pred_y):
    if pred_y >= 0:
        return 1
    else:
        return -1

def linear(pred_y):
        return pred_y

def training():
    if listoflabel[0] == 'Adelie' and listoflabel[1] == 'Gentoo':
        frames = [train_x1, train_x2]
        train_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = train_x1new.loc[:, 'species']  # 60 rows with column species only
        y_actual.replace(['Adelie', 'Gentoo'], [1, -1], inplace=True)
        s = train_x1new.loc[:, listoffeature]  # 60 rows with 2 features
        y = np.array(s)  # 35.9   18.7   bias
        if bias == 1:
            weights = np.random.rand(3)  # 0.2   0.3   0.4
        else:
            weights = np.random.rand(2)
        for e in range(epochs):
            score = 0
            index = 0
            sum=0.0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = y_actual[index] - py
                    v = (eta * loss) * i
                    weights = weights + v
                else:
                   score += 1
                index += 1
            index = 0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = (y_actual[index] - py)
                    sum += loss*loss
                index += 1
            MSE = (sum/60)
            if (MSE< threashold):
                 break
    elif listoflabel[0] == 'Adelie' and listoflabel[1] == 'Chinstrap':
        frames = [train_x1, train_x3]
        train_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = train_x1new.loc[:, 'species']  # 60 rows with column species only
        y_actual.replace(['Adelie', 'Chinstrap'], [1, -1], inplace=True)
        s = train_x1new.loc[:, listoffeature]  # 60 rows with 2 features
        y = np.array(s)
        if bias == 1:
            weights = np.random.rand(3)  # 0.2   0.3   0.4
        else:
            weights = np.random.rand(2)
        for e in range(epochs):
            index = 0
            sum=0.0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = (y_actual[index] - py)
                    v = eta * loss * i
                    weights = weights + v
                index += 1
            index=0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = (y_actual[index] - py)
                    sum += loss*loss
                index += 1
            MSE = (sum / 60)
            if (MSE < threashold):
                break
    else:
        frames = [train_x2, train_x3]
        train_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = train_x1new.loc[:, 'species']  # 60 rows with column species only
        y_actual.replace(['Gentoo', 'Chinstrap'], [1, -1], inplace=True)
        s = train_x1new.loc[:, listoffeature]  # 60 rows with 2 features
        y = np.array(s)
        if bias == 1:
            weights = np.random.rand(3)  # 0.2   0.3   0.4
        else:
            weights = np.random.rand(2)
        for e in range(epochs):
            index = 0
            sum=0.0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = (y_actual[index] - py)
                    v = eta * loss * i
                    weights = weights + v
                index += 1
            index=0
            for i in y:
                if bias == 1:
                    i = np.append(i, bias)
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                else:
                    pred_y = np.dot(weights.transpose(), i)
                    py = linear(pred_y)
                if py != y_actual[index]:
                    loss = (y_actual[index] - py)
                    sum += loss*loss
                index += 1
            MSE = (sum / 60)
            if (MSE < threashold):
                    break
    return weights
def testing(weights):
    global tp
    global tn
    global fn
    global fp
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    if listoflabel[0] == 'Adelie' and listoflabel[1] == 'Gentoo':
        frames = [test_x1, test_x2]
        test_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = test_x1new.loc[:, 'species']  # 40 rows with column species only
        y_actual.replace(['Adelie', 'Gentoo'], [1, -1], inplace=True)
        s = test_x1new.loc[:, listoffeature]  # 40 rows with 2 features
        y = np.array(s)  # 35.9   18.7   bias
        score = 0
        index = 0
        for i in y:
            if bias == 1:
                i = np.append(i, bias)
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            else:
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            if index < 20:
                if py == y_actual[index]:
                    score += 1
                    tp += 1
                else:
                    fp += 1
            else:
                if py == y_actual[index]:
                    score += 1
                    tn += 1
                else:
                    fn += 1
            index += 1
    elif listoflabel[0] == 'Adelie' and listoflabel[1] == 'Chinstrap':
        frames = [test_x1, test_x3]
        test_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = test_x1new.loc[:, 'species'] #40 rows with column species only
        y_actual.replace(['Adelie', 'Chinstrap'], [1, -1], inplace=True)
        s = test_x1new.loc[:, listoffeature]   #40 rows with 2 features
        y = np.array(s)
        index = 0
        score = 0
        for i in y:
            if bias == 1:
                i = np.append(i, bias)
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            else:
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            if index < 20:
                if py == y_actual[index]:
                    score += 1
                    tp += 1
                else:
                    fp += 1
            else:
                if py == y_actual[index]:
                    score += 1
                    tn += 1
                else:
                    fn += 1
            index += 1
    else:
        frames = [test_x2, test_x3]
        test_x1new = pd.concat(frames).reset_index(drop=True)
        y_actual = test_x1new.loc[:, 'species'] #40 rows with column species only
        y_actual.replace(['Gentoo', 'Chinstrap'], [1, -1], inplace=True)
        s = test_x1new.loc[:, listoffeature]   #40 rows with 2 features
        y = np.array(s)
        index = 0
        score = 0
        for i in y:
            if bias == 1:
                i = np.append(i, bias)
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            else:
                pred_y = np.dot(weights.transpose(), i)
                py = signum(pred_y)
            if index < 20:
                if py == y_actual[index]:
                    score += 1
                    tp += 1
                else:
                    fp += 1
            else:
                if py == y_actual[index]:
                    score += 1
                    tn += 1
                else:
                    fn += 1
            index += 1
    #print('Score in testing is ', (score/40)*100)
    return s
def decisionBoundary(weights, testSamples):
    xLabel1 = testSamples.iloc[0:20, 0]
    xlabel2 = testSamples.iloc[0:20, 1]
    yLabel1 = testSamples.iloc[20:40, 0]
    yLabel2 = testSamples.iloc[20:40, 1]
    plt.scatter(xLabel1, xlabel2, c="blue")
    plt.scatter(yLabel1, yLabel2, c="red")
    if bias:
        xp1 = (- (weights[0] * testSamples.min()[0]) - weights[2]) / weights[1]  # assumed x2= by the minimum value in feature 1
        xp2 = (- (weights[0] * testSamples.max()[0]) - weights[2]) / weights[1]  # assumed x2=by the maximum value in feature 1
    else:
        xp1 = (- (weights[0] * testSamples.min()[0])) / weights[1]
        xp2 = (- (weights[0] * testSamples.max()[0])) / weights[1]
    x = [testSamples.min()[0], testSamples.max()[0]]
    y = [xp1, xp2]
    plt.plot(x, y)
    plt.show()
def confusion_Matrix():
    confusionMatrix = np.array([[tp, fp], [fn, tn]])
    print(confusionMatrix)
    print("Overall accuracy is ", ((tp+tn)/40)*100)
    if tp+fn == 0:
        recall = 0
    else:
        recall = (tp/(tp+fn))
    print("Recall is ", recall)
    if tp+fp == 0:
        precision = 0
    else:
        precision = (tp/(tp+fp))
    print("Precision is", precision)
weights = training()
s = testing(weights)
confusion_Matrix()
decisionBoundary(weights, s)