import cv2
import numpy as np
import os

### Load Training Data

datapath = './data/'
labels = []
class_id = 0
dataset = []
names = {}

for fx in os.listdir(datapath):
	if fx.endswith('.npy'):
		names[class_id] = fx[0:-4] 
		data = np.load(datapath+fx)
		#print(dataset.shape)
		target = class_id * np.ones((data.shape[0],))
		class_id += 1
		dataset.append(data) 
		labels.append(target)

training_X = np.concatenate(dataset, axis = 0)
training_Y = np.concatenate(labels , axis = 0).reshape((training_X.shape[0], -1))
#print(training_X.shape, training_Y.shape)

training_data = np.concatenate((training_X, training_Y), axis = 1)
#print(training_data.shape)

###

### KNN for face classification

def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k = 5):
	dist = []
	for i in range(train.shape[0]):
		ix = train[i, :-1]
		iy = train[i, -1]
		dist.append([distance(test, ix), iy])

	dist = sorted(dist, key = lambda f: f[0])[:k]
	l = np.array(dist)[:, -1]
	output = np.unique(l, return_counts = True)
	index = np.argmax(output[1])
	return output[0][index]

###

### Capture testing data from webcam


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized


capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./third-party/haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('./third-party/haarcascade_mcs_nose.xml')
mustache = cv2.imread('mustache.png',-1)


while True:
	#capturing frame
	ret,frame = capture.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
	# capturing all faces in the frame
	faces = face_cascade.detectMultiScale(frame, 1.4, 5)

	# iterating over all the faces
	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
		offset = 10
		face_section_color = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section_gray = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
		roi_gray = cv2.resize(face_section_gray, (100,100))
		testing_data = roi_gray.flatten()
		pred_name = names[int(knn(training_data, testing_data))]
		cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

		nose = nose_cascade.detectMultiScale(face_section_gray, scaleFactor=1.5, minNeighbors=5)
			
		for (nx, ny, nw, nh) in nose:
			#cv2.rectangle(face_section_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
			roi_nose = face_section_gray[ny: ny + nh, nx: nx + nw]
			mustache2 = image_resize(mustache.copy(), width=nw)

			mw, mh, mv = mustache2.shape
			for i in range(0, mw):
				for j in range(0, mh):
					if mustache2[i, j][3] != 0: # alpha 0
						face_section_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]

	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
	cv2.imshow('Project', frame)
	key_presssed = cv2.waitKey(1) & 0xFF
	if key_presssed == ord('e'):
		break

capture.release()
cv2.destroyAllWindows()

###
