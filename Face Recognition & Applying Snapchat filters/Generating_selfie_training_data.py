import cv2
import numpy as np

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./third-party/haarcascade_frontalface_default.xml')
face_data = []
skip = 0
datapath = './data/'
filename = input('Enter the name of the person ')

while True:

	ret,frame = capture.read()

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.4, 5)
	faces = sorted(faces, key = lambda f: f[2] * f[3], reverse = True)
	isFirst = True

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 2)
		skip += 1

		if isFirst and skip%10 == 0:
			isFirst = False
			offset = 10
			face_section_color = frame[y-offset:y+h+offset, x-offset:x+w+offset]
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			face_section_gray = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
			face_section = cv2.resize(face_section_gray, (100,100))
			face_data.append(face_section)
			print(len(face_data))
			cv2.imshow('Captured img', face_section_color)
			
	cv2.imshow('Original', frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('e')):
		break

face_data = np.asarray(face_data)
print(face_data.shape)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

np.save(datapath+filename+'.npy', face_data)
print('Data successfully saved at '+datapath+filename+'.npy')

capture.release()
cv2.destroyAllWindows()
