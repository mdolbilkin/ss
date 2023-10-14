import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from PIL import Image
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

height = 500
width = 500

def image_preprocessing_2bw(pathToImage):
    out_img_dim = (height, width)
    lower_green = np.array([25, 15, 15])
    upper_green = np.array([85, 255, 255])

    image = cv2.imread(pathToImage)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        cnt = max(contours, key=cv2.contourArea)
        brect = cv2.boundingRect(cnt)
        x,y,w,h = brect
        cropped_image = mask[y:y+h, x:x+w]
        cv2.rectangle(mask, brect, (255,255,255), 1)
    
    processed_image = cv2.bitwise_not(cropped_image)
    output_image = cv2.resize(processed_image, out_img_dim, interpolation = cv2.INTER_AREA)
    return(output_image)

def image_preprocessing_2color(pathToImage):
    out_img_dim = (height, width)
    lower_green = np.array([25, 15, 15])
    upper_green = np.array([85, 255, 255])

    image = cv2.imread(pathToImage)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        cnt = max(contours, key=cv2.contourArea)
        brect = cv2.boundingRect(cnt)
        x,y,w,h = brect
        cropped_image = image[y:y+h, x:x+w]
        cv2.rectangle(mask, brect, (255,255,255), 1)

    output_image = cv2.resize(cropped_image, out_img_dim, interpolation = cv2.INTER_AREA)
    return(output_image)

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval = 150,
    brightness_range=[0.7, 1.3]
)
path = os.listdir("Data/OriginData")
for dict in path:
    path1 = "Data/OriginData\\" + dict
    for photo in os.listdir(path1):
        pic = load_img(path1 + "\\" + photo)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1, ) + pic_array.shape) 
        count = 0
        for batch in datagen.flow(pic_array, batch_size=64,save_to_dir="Data/AUG2C\\" + dict, save_prefix=photo[:photo.find(".")], save_format='jpg', shuffle=False):
            count += 1
            if count > 50:
                break
        print(path1 + "\\" + photo)

path = os.listdir("Data/AUG2C")
for dict in path:
    path1 = "Data/AUG2C\\" + dict
    for photo in os.listdir(path1):
        im = Image.open(path1 + "\\" + photo)
        if im.height < im.width:
            rotated_image2 = im.transpose(Image.ROTATE_90)
            rotated_image2 = rotated_image2.resize((750, 1333))
        else:
            rotated_image2 = im
            rotated_image2 = rotated_image2.resize((750, 1333))
        rotated_image2.save(path1 + "\\" + photo)
        im.close()

for dict in path:
    for f in os.listdir('Data/AUG2C/' + dict):
        cv2.imwrite("Data/AUG2BW/"+dict+"/"+f, image_preprocessing_2bw('Data/AUG2C/'+dict+'/'+f))
     
for dict in path:
    for f in os.listdir('Data/AUG2C/' + dict):
        cv2.imwrite("Data/AUG2C/"+dict+"/"+f, image_preprocessing_2color('Data/AUG2C/'+dict+'/'+f))