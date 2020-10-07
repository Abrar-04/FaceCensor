import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
import os

root=Tk()
root.title("Face Censor APP")
root.geometry("500x250") 
root.resizable(0,0)
root.configure(bg='black')

def open_img():
    global my_image
    root.filename=filedialog.askopenfilename(initialdir='/media/abrar/DATA/projects/Face Censor',title = "Select a File", filetypes = (("all files", "*.*"),("Text files", "*.txt*")))
    my_label=Label(root,text=root.filename).pack()
    my_image=cv2.imread(root.filename)
    
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_img=my_image.copy()
    roi=my_image.copy()
    face_rects=face_cascade.detectMultiScale(face_img,scaleFactor=1.3,minNeighbors=3)
    
    for (x,y,w,h) in face_rects:
        roi=roi[y:y+h,x:x+w]
        blurred_roi=cv2.medianBlur(roi,35)
        
        face_img[y:y+h,x:x+w]=blurred_roi
    
    result=face_img
    path = '/media/abrar/DATA/projects/Face Censor/output'
    cv2.imwrite(os.path.join(path , 'blurred.jpg'),result)
    messagebox.showinfo(title='Face ', message='Check Output folder')

my_btn=Button(root,text='Browse:',command=open_img,height=5,width=15,font=('Comic Sans MS',25,'bold'),bg='DarkOrchid3',fg='Yellow').pack()

root.mainloop()