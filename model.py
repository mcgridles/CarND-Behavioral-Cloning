import csv
import os
import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# GLOBALS
csv_path = '../data/driving_log.csv'
img_dir_path = '../data/IMG'
correction = 0.2

def processImage(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def dataGenerator(samples, batch_size=64):
	"""
	Generator for data batches containing images for input and steering angles
	for output.
	"""
	num_samples = len(samples)

	while True:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				# generate path and read images
				img_path_c = os.path.join(img_dir_path, os.path.basename(batch_sample[0]))
				img_path_l = os.path.join(img_dir_path, os.path.basename(batch_sample[1]))
				img_path_r = os.path.join(img_dir_path, os.path.basename(batch_sample[2]))

				image_center = processImage(cv2.imread(img_path_c))
				image_left = processImage(cv2.imread(img_path_l))
				image_right = processImage(cv2.imread(img_path_r))

				# read steering angles and correct for camera location
				steering_center = float(batch_sample[3])
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				images.append(image_center)
				images.append(image_left)
				images.append(image_right)
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)

			# augment images + angles by flipping
			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_images.append(cv2.flip(image, 1))
				augmented_angles.append(angle)
				augmented_angles.append(-angle)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def train(train_generator, valid_generator, num_train, num_valid):
	"""
	Trains the model using the data generators. The model is built with Keras
	using a modified LeNet architecture based on the network used for the
	traffic sign classifier project.
	"""
	model = Sequential()

	# pre-processing
	model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160,320,3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))

	# convolutional layers
	model.add(Convolution2D(8, 1, 1, activation='relu'))
	model.add(Convolution2D(16, 5, 5, activation='relu'))
	model.add(MaxPooling2D(border_mode='same'))
	model.add(Convolution2D(32, 5, 5, activation='relu'))
	model.add(MaxPooling2D(border_mode='same'))

	# fully connected layers
	model.add(Flatten())
	model.add(Dropout(0.7))
	model.add(Dense(120, activation='relu'))
	model.add(Dense(84, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator,
						samples_per_epoch=num_train,
						validation_data=valid_generator,
						nb_val_samples=num_valid,
						nb_epoch=3)

	model.save('model.h5')

def main():
	samples = []
	with open(csv_path, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			samples.append(row)

	# create generators
	train_samples, valid_samples = train_test_split(samples, test_size=0.2)
	train_generator = dataGenerator(train_samples, 64)
	valid_generator = dataGenerator(valid_samples, 64)

	# train model
	train(train_generator, valid_generator, len(train_samples), len(valid_samples))

if __name__ == '__main__':
	main()
