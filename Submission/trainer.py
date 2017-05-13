import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from functions import extract_features
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Read in cars and notcars
right = glob.glob('./vehicles/GTI_Right/image*.png')
left = glob.glob('./vehicles/GTI_Left/*.png')
far = glob.glob('./vehicles/GTI_Far/*.png')
Middle = glob.glob('./vehicles/GTI_MiddleClose/*.png')
kitti = glob.glob('./vehicles/KITTI_extracted/*.png')

extras = glob.glob('./non-vehicles/Extras/*.png')
gti = glob.glob('./non-vehicles/GTI/*.png')

notcars = np.concatenate((gti, extras))
cars = np.concatenate((right, left, far, Middle))

cars = shuffle(cars)
notcars = shuffle(notcars)
sample_size = 1900
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

color_space = 'YUV' # Can be RGB, HSV, LUV, LAB, HLS, YUV, YCrCb
orient = 16  # HOG orientations
pix_per_cell = 6 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' #'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off LAB
hist_feat = True # Histogram features on or off HSV
hog_feat = True # HOG features on or off YUV

print('Extracting Car Features...')
car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

print('\nExtracting NotCar Features...')

notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
scaled_X, y, test_size=0.2, random_state=rand_state)

print('\nUsing:',orient,'orientations',pix_per_cell,
'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
print('\nFitting...')
svc = LinearSVC()
#mlp = MLPClassifier()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
#mlp.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
#  Check the score of the SVC
print('\nTesting...')
score = round(svc.score(X_test, y_test), 4)
print('Test Accuracy of SVC = ', score)
# Check the prediction time for a single sample
t=time.time()

#plk_dict = {}
#plk_dict['clf'] = svc
#plk_dict['scaler'] = X_scaler
#pickle.dump(plk_dict, open('./trainedSVC.p', "wb"))
