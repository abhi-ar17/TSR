import cv2
import tensorflow as tf

DATADIR = "fdtju/datasets"

CATEGORIES =["70","50","Cattle","Narrow","No_left","No_Parking","No_right","No_stop","Pedestrian","school","Stop","Two" ,"Hump", "Men","Noentry"]

def prepare(filepath):
	IMG_SIZE=64
	img_array = cv2.imread(filepath)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)

model=tf.keras.models.load_model("model_categorical_complex.model")

prediction = model.predict([prepare('bcd.jpg')])

print(prediction)
