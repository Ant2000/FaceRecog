import numpy as np
import cv2
import pickle
import os
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import shutil
import time

#Local Modules
from common import clock, draw_str



def AddFace(namex):	
	#create dataset
	total=0
	def get_image():
		im = cam.read()
		return im
	def detect(img, cascade):
		rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
		if len(rects) == 0:
			return []
		rects[:,2:] += rects[:,:2]
		return rects
	def draw_rects(img, rects, color):
		for x1, y1, x2, y2 in rects:
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
	test=0
	cascade = cv2.CascadeClassifier("D:\OpenCV\opencv\data\haarcascades_cuda\haarcascade_frontalface_alt.xml")
	nested = cv2.CascadeClassifier("D:\OpenCV\opencv\data\haarcascades_cuda\haarcascade_eye.xml")
	cam= VideoStream(src=0).start()
	total=0
	try:
		os.mkdir("./dataset/"+namex)
	except:
		shutil.rmtree("./dataset/"+namex)
		os.mkdir("./dataset/"+namex)
	while True:
		img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		t = clock()
		rects = detect(gray, cascade)
		vis = img.copy()
		draw_rects(vis, rects, (0, 255, 0))
		if not nested.empty():
			for x1, y1, x2, y2 in rects:
				roi = gray[y1:y2, x1:x2]
				vis_roi = vis[y1:y2, x1:x2]
				subrects = detect(roi.copy(), nested)
				draw_rects(vis_roi, subrects, (255, 0, 0))
				print("Taking image")
				camera_capture = get_image()
				xfile = "./dataset/"+namex+"/"+str(total)+".jpg"
				cv2.imwrite(xfile, camera_capture)
				total=total+1
		dt = clock() - t
		draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
		cv2.imshow('facedetect', vis)
		if(total>5):
			break
		if cv2.waitKey(5) == 27:
			break
	cv2.destroyAllWindows()
	#extract embeddings
	dataset="./dataset"
	embeddings="./output/embeddings.pickle"
	detector="./face_detection_model"
	model="./openface_nn4.small2.v1.t7"
	conf=0.7
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([detector, "deploy.prototxt"])
	modelPath = os.path.sep.join([detector,"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(model)
	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(dataset))
	knownEmbeddings = []
	knownNames = []
	total = 0
	for (i, imagePath) in enumerate(imagePaths):
		print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		if len(detections) > 0:
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]
			if confidence > conf:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1
	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open(embeddings, "wb")
	f.write(pickle.dumps(data))
	f.close()
	
	#train model
	embedding="./output/embeddings.pickle"
	recogniser="output/recognizer.pickle"
	le1="./output/le.pickle"
	print("[INFO] loading face embeddings...")
	data = pickle.loads(open(embedding, "rb").read())
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)
	f = open(recogniser, "wb")
	f.write(pickle.dumps(recognizer))
	f.close()
	f = open(le1, "wb")
	f.write(pickle.dumps(le))
	f.close()
	
	#recognise face
	detector="./face_detection_model"
	model="./openface_nn4.small2.v1.t7"
	recogniser="./output/recognizer.pickle"
	le1="./output/le.pickle"
	conf=0.9
	test=0
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([detector, "deploy.prototxt"])
	modelPath = os.path.sep.join([detector,"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(model)
	recognizer = pickle.loads(open(recogniser, "rb").read())
	le = pickle.loads(open(le1, "rb").read())
	print("[INFO] starting video stream...")
	time.sleep(4.0)
	fps = FPS().start()
	while True:
		frame = cam.read()
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > conf:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW < 20 or fH < 20:
					continue
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
				if(name==namex):
					test=test+1
				else:
					test=test-1
				print(test)
		fps.update()
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if(test==3):
			print("Thank You")
			auth=1
			break
		if(test==-3):
			print("Not Authoised")
			auth=0
			break
		if key == ord("q"):
			break
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	cv2.destroyAllWindows()
	cam.stop()
	return(auth)
a=input("Enter name: ")
AddFace(a)
