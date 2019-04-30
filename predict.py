import cv2
import tensorflow as tf

DATADIR = "datasets"

CATEGORIES =["70","50","Cattle","Narrow","No_left","No_Parking","No_right","No_stop","Pedestrian","school","Stop","Two" ,"Hump", "Men","Noentry"]

def prepare(filepath):
	IMG_SIZE=200
	img_array = cv2.imread(filepath)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model=tf.keras.models.load_model("trafficsign.model")

prediction = model.predict([prepare('asd.jpg')])

print(int(prediction[0][0]))
