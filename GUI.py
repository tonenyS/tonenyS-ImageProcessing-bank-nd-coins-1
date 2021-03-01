from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
import mahotas as mh
import math
import numpy as np
import argparse
import glob
import cv2
import pickle
import requests
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

page1 = None
page2 = None
loaded_model = pickle.load(open('Banknote_Model.sav', 'rb'))


root = Tk()
root.option_add("*Font", "consolas 13")
root.title("Currency Converter")
root.iconbitmap("C:/Users/HP/Desktop/1000%/Project/gui/euicon.ico")
root.minsize(350,150) 
Label(root, text="Welcome To Currency converter Program").pack()


def bank_window():
    global page1
    page1 = Toplevel() 
    page1.title('Banknote Converter')
    page1.iconbitmap("C:/Users/HP/Desktop/1000%/Project/gui/euicon.ico")
    page1.minsize(500,500)  
    Label(page1, text="Please choose a image of banknotes to convert for currency").pack()
    op_imgbankpage = Button(page1, text = "Selelct Image", command=selectimage1).pack()
    backmenu_btn = Button(page1, text = "Back To Menu", command=page1.destroy).pack(side="bottom")
       
    

    page1.mainloop()


def coin_window():
    global page2
    page2 = Toplevel()
    page2.title('Coin Converter')
    page2.iconbitmap("C:/Users/HP/Desktop/1000%/Project/gui/euicon.ico")
    page2.minsize(500,500) 
    Label(page2, text="Please choose a image of coins to convert for currency").pack()
    op_imgbankpage = Button(page2, text = "Selelct Image",command=selectimage2).pack()
    backmenu_btn = Button(page2, text = "Back To Menu", command=page2.destroy).pack(side="bottom")
    

    page2.mainloop()



def selectimage1():
    bank_dict = {'Five':5,'Ten':10,'Twenty':20,'Fifty':50,'Onehundred':100,'Twohundred':200,'Fivehundred':500}
    url = 'https://api.exchangerate-api.com/v4/latest/EUR'
    global my_image
    page1.filename = filedialog.askopenfilename(initialdir="‪C:/Users/HP/Desktop/1000%/Project/gui/images", title="Select File",filetypes=(("jpg files", "*jpg"),("all files", "*.*")))
    my_image = ImageTk.PhotoImage(Image.open(page1.filename)) 
    Label(page1,image=my_image).pack()
    results = predictMaterialBanknote(page1.filename)
    response = requests.get(url)
    data = response.json()
    EUR_RATE = data['rates']['EUR']
    THB_RATE = data['rates']['THB']
    RESULTS_BALANCE = int(bank_dict.get(results, None))
    RESULTS_BALANCE *= THB_RATE
    Label(page1, text="Result of convert to (THB): %.2f"%RESULTS_BALANCE+" BATH [%.4f]"%THB_RATE).pack(side="bottom") #แปลงเงินยูโรเป็นเงินไทย
    Label(page1, text="Result(EUR):"+results).pack(side="bottom") #จำนวนเงินยูโร
    

def selectimage2():
    global my_image
    page2.filename = filedialog.askopenfilename(initialdir="‪C:/Users/HP/Desktop/1000%/Project/gui/test", title="Select File",filetypes=(("jpg files", "*jpg"),("all files", "*.*")))
    detech_image,total = CoinDetech(page2.filename)
    # cv2.imshow("Coin Converter",detech_image)
    url = 'https://api.exchangerate-api.com/v4/latest/EUR'
    response = requests.get(url)
    data = response.json()
    EUR_RATE = data['rates']['EUR']
    THB_RATE = data['rates']['THB']
    RESULTS_BALANCE = total
    RESULTS_BALANCE *= THB_RATE
    Label(page2, text="Result(EUR): {} Euro (THB): {} Bath".format(total,round(RESULTS_BALANCE,2))).pack(side="bottom") #จำนวนเงินยูโร
    cv2.imshow("detech",detech_image)


#*************************Buttonmenu*************************

open_bankbtn = Button(root, text = "Click to Convert Banknote", command=bank_window)
open_bankbtn.pack()

open_coinbtn = Button(root, text = "Click to Convert Coin", command=coin_window)
open_coinbtn.pack()

exist_btn = Button(root, text = "Exist Program", command=quit)
exist_btn.pack()

#************************************************************


class Enum(tuple): __getattr__ = tuple.index


Material = Enum(('Five','Ten','Twenty','Fifty','Onehundred','Twohundred','Fivehundred'))
 # กำหนดชนิดของคลาสที่จะใช้เทรน
clf = MLPClassifier(solver="lbfgs")


def predictMaterialBanknote(filename):
    image = mh.imread(filename)
    image = mh.colors.rgb2gray(image, dtype=np.uint8) 
    # resize image while retaining aspect ratio
    # d = 1024 / image.shape[1]
    # dim = (1024, int(image.shape[0] * d))
    # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(image)
    ft = mh.features.haralick(img).ravel()
    s = loaded_model.predict([ft])
    return Material[int(s)]

def caculateRtoCoint(rate):
    if rate <= 42:
    	return	"1 cent",1   
    elif rate <= 47 and rate >= 43:
    	return	"2 cent",2
    elif rate >= 48 and rate <= 50:
    	return	"10 cent",10
    elif rate <= 53 and rate >= 51:
    	return	"5 cent",5
    elif rate >= 54 and rate <= 56:
    	return	"20 cent",20
    elif rate >= 58 and rate <= 61:
    	return	"50 cent",50
    elif rate <= 58 and rate >= 57:
    	return "1 euro",100
    elif rate >= 62 and rate <= 65:
    	return	"2 euro",200
    return None,0
 

def CoinDetech(path):
    total = 0
    image = cv2.imread(path)
    d = 1024 / image.shape[1]
    dim = (1024, int(image.shape[0] * d))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    output = image.copy()
    # thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    # cv2.imshow("gray",)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    circles	= cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT, dp=2.0, minDist=93,param1=200, param2=100, minRadius=1, maxRadius=90)
    circles	= np.uint16(np.around(circles))

    for	(x,y,r) in circles[0,:]:
        results,_numeric = caculateRtoCoint(r)
        total = total + int(_numeric)
        #	draw	the	outer	circle
        cv2.circle(image,(x,y),r,(0,255,0),2)
        #	draw	the	center	of	the	circle
        cv2.putText(image,"",(x,y), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(image,"{}".format(results),(int(x-(r/2)),y), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0),thickness=2,lineType=cv2.LINE_AA)
    return image,total/100

root.mainloop()

