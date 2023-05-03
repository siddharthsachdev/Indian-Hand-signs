import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import imagePreprocessingUtils as ipu


filename = input('Enter the csv file name to read: ')
sub = pd.read_csv(filename)
y_pred = np.array(sub.pop('PredictedLabel'))
y_test = np.array(sub.pop('TrueLabel'))

class_labels = ipu.get_labels()

