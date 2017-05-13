import cv2
import glob
import pickle
import numpy as np
from functions import *
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

class VehicleTracker:
    def __init__(self):
        self.heatmap_list = []
        self.list = []
        self.heatmap = None
        self.svc = None
        self.X_scaler = None
        self.orient = None
        self.pix_per_cell = None
        self.cell_per_block = None
        self.spatial_size = None
        self.hist_bins = None
        self.Y_min_max = []
        self.scales = []
        self.smoothing = None

    def set_model(self):
        dict_pickle = pickle.load(open("./trainedSVC.p", "rb"))

        self.svc = dict_pickle['clf']
        self.X_scaler = dict_pickle['scaler']

        self.orient = 16  # HOG orientations

        self.pix_per_cell = 6 # HOG pixels per cell

        self.cell_per_block = 2 # HOG cells per block

        self.spatial_size = (64, 64) # Spatial binning dimensions

        self.hist_bins = 32    # Number of histogram bins

        self.Y_min_max = [388, 656]

        self.scales = [0.75, 1, 1.5, 2]

        self.smoothing = 23

    def video_pipline(self, img):
        aimg = img.astype(np.float32)/255

        self.heatmap = np.zeros_like(img[:,:,0])

        heatmap_sum = 0

        for scale in self.scales:
            img_boxes = find_cars(aimg, self.Y_min_max[0], self.Y_min_max[1], scale,
                self.svc, self.X_scaler,self.orient, self.pix_per_cell, self.cell_per_block,
                self.spatial_size, self.hist_bins)
            self.heatmap = add_heat(self.heatmap, img_boxes)

        self.heatmap_list.append(self.heatmap)

        if len(self.heatmap_list) > self.smoothing:
            del(self.heatmap_list[0])

        for heatmap in self.heatmap_list:
            heatmap_sum += heatmap

        heatmap_sum = np.clip(heatmap_sum, 0, 255)

        self.heatmap = apply_threshold(heatmap_sum, 28)

        labels = label(self.heatmap)

        final = draw_labeled_bboxes(np.copy(img), labels)
        return final

Tracker = VehicleTracker()
Tracker.set_model()

#project_vid = 'project_video.mp4'
test_vid = 'test_video.mp4'

clip1 = VideoFileClip(test_vid)
white_clip = clip1.fl_image(Tracker.video_pipline)
white_clip.write_videofile('test_video_done2.mp4', audio=False)





