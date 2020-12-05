import numpy as np
import os
import cv2

def crop_img(file_obj, image):
    co_ord_list = file_obj.readlines()
    #print(co_ord_list)
    x1 = y1 = np.inf
    x2 = y2 = -1*np.inf
    for x_y in co_ord_list:
        a = x_y.split(' ')   
        x_min = np.floor(float(a[4]))
        y_min = np.floor(float(a[5]))
        x_max = np.ceil(float(a[6]))
        y_max  = np.ceil(float(a[7]))
        if(x_min < x1):
            x1 = x_min
        if(y_min < y1):
            y1 = y_min
        if(x_max > x2):
            x2 = x_max
        if(y_max > y2):
            y2 = y_max
    print(x1,x2,y1,y2)
    if(x1-2>=0):
        x1 -= 2
    if(y1-2>=0):
        y1-=2
    if(x2+2 <= image.shape[0]):
        x2+=2
    if(y2+2 <= image.shape[1]):
        y2 += 2
    crop = image[int(x1): int(x2), int(y1):int(y2)]
    return crop

img_path = "image_2"
label_path = "label_2"
imgs = os.listdir(imag_path)
imgs.sort()
labels = os.listdir(label_path)
labels.sort()
crop_image_path = "new_image"
i = 0
for label in labels:
	my_label = open(os.path.join(label_path,label),'r')
	temp_img = label.split('.')[0] + '.png'
	my_image = cv2.imread(os.path.join(img_path, temp_img))
	crop_img = crop_img(my_label, my_image)
	if(crop_img.size):
		write_path = os.path.join(crop_image_path,temp_img)
		cv2.imwrite(write_path, crop_img)
	if(i==2):
		break


