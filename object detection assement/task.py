import cv2
import numpy as np
import matplotlib.pyplot as plt


#read images

car1=cv2.imread('object detection assement/data/Rapid-red-front-rus.jpg')
car1=cv2.cvtColor(car1,cv2.COLOR_BGR2RGB)



car2=cv2.imread('object detection assement/data/bbb.png')
car2=cv2.cvtColor(car2,cv2.COLOR_BGR2RGB)

car3=cv2.imread('object detection assement/data/rus.jpg')
car3=cv2.cvtColor(car3,cv2.COLOR_BGR2RGB)


#display image function
def dispaly_image(img,title,cmap=None):

    fig=plt.figure(figsize=(12,10))

    ax=fig.add_subplot(111)

    ax.imshow(img,cmap=cmap)
    plt.title(title)
    plt.show()

dispaly_image(car1,'car')



#load traind_pre file xml

casceds=cv2.CascadeClassifier('object detection assement/data/haarcascade_russian_plate_number.xml')

#create detect function and draw rectangle and blur car plate

def detect_car_plate(img):

    #craete copy from orignal 
    img_copy=img.copy() #to draw rectangle 
    blured_plate=img.copy() #to blur the plate
    #get corners postion for draw and blur 
    corners_rec=casceds.detectMultiScale(img_copy,minNeighbors=7) 
   
    for (x,y,w,h) in corners_rec:

        
        #crate a ROI 
        
        roi=img[y:y+h,x:x+w]

        #drawen2=cv2.rectangle(img_copy[y:y+h,x:x+w],(x,y),(x+w,y+h),color=(255,0,0),thickness=4)
        
        #draw rectangle on plate 

        detected_plate=cv2.rectangle(img_copy,(x,y),(x+w,y+h),color=(255,0,0),thickness=4)
        
        #blur ori area
        blured_roi=cv2.medianBlur(roi,25)

        #reassigment arrays

        blured_plate[y:y+h,x:x+w]=blured_roi


    return detected_plate,blured_plate




plate1,car1_blured=detect_car_plate(car1)

dispaly_image(plate1,'the detected plate for car1')


dispaly_image(car1_blured,'blur car plate')


plate2,car2_blured=detect_car_plate(car2)


dispaly_image(plate2,'the detected plate for car2')


dispaly_image(car2_blured,'blur car plate')


plate3,car3_blured=detect_car_plate(car3)


dispaly_image(plate3,'the detected plate for car2')


dispaly_image(car3_blured,'blur car plate')
















