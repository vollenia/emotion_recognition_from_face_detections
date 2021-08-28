# Importing modules
import os
import argparse
import time
import pandas as pd
import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True,
	help = "path to the data file directory")
parser.add_argument("-l", "--labels", required=True,
	help = "path to the labels")
parser.add_argument("-f", "--frames", required=True, type=int,
	help = "Nr of frames to be extracted")
args = vars(parser.parse_args())

# Creating a list with paths from directory
list_of_files = sorted([os.path.join(args["data"], i) for i in os.listdir(args["data"])])

# Defining face detector
detector = dlib.get_frontal_face_detector()

# Extracting features & reducing the frames
# Target frames
def frame_selection(frames, nr):
	vip_frames = np.array([frame for frame in frames[:len(frames)//nr*nr:len(frames)//nr]])
	
	return vip_frames # returns a 3D array

data = {}       
def process_video(files, crop=100):
	start_time = time.time()
	for file_path in files:
		cap = FileVideoStream(file_path).start()
		frames = []
		while cap.more():
			try:
				frame = cap.read()
				# Reduces the res of the file for faster processing 
				frame = imutils.resize(frame, width=400)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				rects = detector(frame, 0)
				if len(rects) > 0:  # if at least one face detected
					face = rects[0]
					x1, y1 = face.left(), face.top()
					x2, y2 = face.right(), face.bottom()
					crop_img = frame[y1:y2, x1:x2]
					try:
						crop_img = cv2.resize(crop_img, (crop, crop))
					except Exception:
						print(f'crop/resize failed for {file_path} for bounding box with coordinates {x1, y1, x2, y2}')
					else:
						frames.append(crop_img)
			except AttributeError:
				pass
		cap.stop()
		if len(frames) >= args["frames"]: # checks if enough frames were accumulated
			data[file_path.split('/')[-1].replace('.avi', '')] = frame_selection(frames, args["frames"])# nr of target frames
		else:
			data["Y"+file_path.split('/')[-1].replace('.avi', '')] = None
	end_time = time.time()
	duration = end_time - start_time
	print(f'video processing took {duration} seconds')

process_video(list_of_files, 100)

# Extracting labels
ref = ["anger", "happiness", "neutral", "sadness"]
labels = {}
def get_labels(file):
	start_time = time.time()
	nr_l = 0
	nr_bl = 0
	with open(file, 'r') as f:
		for line in f:
			inner_d = {}
			splt = line.split(';')
			if splt[1] in ref: # filtering down to 4 calsses
				inner_d["LABEL"] = splt[1]
				inner_d["ACTIVATION"] = float(splt[2][2:])
				inner_d["VALENCE"] = float(splt[3][2:])
				labels[splt[0]] = inner_d
			else:
				labels["X"+splt[0]] = None
				nr_bl +=1
			nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'label processing of {nr_l} line(s) ({nr_bl} bad labels detected) took {duration} seconds')

get_labels(args["labels"])

# Creating dataset as dictionary
dataset_d = {}
def create_dataset(labels, data):
	start_time = time.time()
	nr_l = 0
	nr_bl = 0
	for x, y, i, j  in zip(labels.keys(), data.keys(), labels.values(), data.values()):
		if x == y:
			features = {}
			features["FEATURES"]= j
			i.update(features)
			dataset_d[x] = i
		else:
			print(f'filtered out: {x} {y}')
			nr_bl += 1
		nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'combining of {nr_l} line(s) ({nr_bl} lines filtered) took {duration} seconds')
create_dataset(labels, data)

# Creating pandas dataframe from dictionary
dataset_df = pd.DataFrame.from_dict(dataset_d, orient='index')

# Storing the dataframe
dataset_df.to_pickle("PATH"+"df_hog_fast"+str(args["frames"])+".pkl")
