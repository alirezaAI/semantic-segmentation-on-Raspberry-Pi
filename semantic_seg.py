""" this code is heavily based on the DeepLab_TFLite_ADE20k.ipynb
https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_ADE20k.ipynb#scrollTo=GzAbm9a9ljRP

"""
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd

import tflite_runtime.interpreter as tflite


# Load the TFLite model and allocate tensors.############################################
interpreter = tflite.Interpreter(model_path="lite-model_deeplabv3-xception65-ade20k_1_default_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print(input_details)

output_details = interpreter.get_output_details()
output_details
#print(input_details)

input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
##########################################################################################
pil_image=Image.open("street.jpg")

new_img=pil_image.resize((input_size[0],input_size[1]))# 513,513

np_new_img = np.array(new_img)
np_new_img = (np_new_img/255).astype('float32')#astype('float32')

np_new_img = np.expand_dims(np_new_img, 0)

interpreter.set_tensor(input_details[0]['index'], np_new_img)
############################################################################################
start = time.time()
interpreter.invoke()
# Retrieve the raw output map.
raw_prediction = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
# Post-processing: convert raw output to segmentation output
seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
seg_map = np.asarray(Image.fromarray(seg_map).resize(pil_image.size))
########################################################################

# by changing the value of the label you can extract any desired class with in the image.
label=7 # 7 based on the csv object class is raod
result=np.where(seg_map == label,255,0) 

PIL_image = Image.fromarray(np.uint8(result)).convert('L') # L (Luminace) converts our result inot a grayscale image
###########################################################################
    		

cv_image=np.array(PIL_image)   # converting the PIL image to CV format if you want to perform any further image processing anlysis using opencv

blur = cv2.medianBlur(cv_image, 13) # applying medianBlur to smoothen the jagged edges

#cv2.imwrite("seg_mask.jpg",blur) # to store the image
 	
cv2.imshow("blured_mask", blur)
cv2.waitKey(0)
    	

end = time.time()
print("it took:" + str(end - start))



























