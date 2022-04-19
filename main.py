import numpy as np

from keras.preprocessing import image
from keras.applications import resnet

#load Keras resnet model pre trained against ImageNet database
model = resnet.ResNet50()

#load the image and resize to 224,224 as required by this model
img = image.load_img('img.png',target_size=(224,224))

#convert image to numpy array
img_array=image.img_to_array(img)

#add 4th dimension since keras expects list of images
img_array=np.expand_dims(img_array,axis=0)

#scale input image to the range used in trained network
img_array=resnet.preprocess_input(img_array)

#make prediction
predictions=model.predict(img_array)

#get names of predicted classes
pred_class=resnet.decode_predictions(predictions,top=5)
for imageid,name,chances in pred_class[0]:
    print("{}:{:2f} chance ".format(name,chances))

