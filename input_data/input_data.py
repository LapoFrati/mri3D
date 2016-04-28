import cPickle as cp
import numpy as np
import cv2

class input_data:

    def __init__(self):
         pass

    @property
    def inputs(self):
        return self.inputs

    @property
    def labels(self):
        return self.labels

    def read_data_sets(self):
         dirdata = "/home/mario/Downloads/DB-AD-CTRL.pkl"
         with open(dirdata, "rb") as fp:
             data = cp.load(fp)
             self.inputs = np.array(data["xs"])
             temp = map( lambda x : [1,0] if x == -1 else [0,1] , data["ys"])
             print temp
             self.labels = np.array(temp)
             image = self.inputs[0][:5202]
             image = image.reshape(17,17,18)
             print image.shape
             cv2.imshow('mri', image[:,:,8])
             cv2.waitKey(0)


data = input_data()
data.read_data_sets()
