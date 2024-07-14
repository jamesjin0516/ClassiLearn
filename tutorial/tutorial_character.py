import joblib
import numpy as np


charmat_low = np.loadtxt("low_emotions.txt")
charmat_high = np.loadtxt("high_emotions.txt")
charmat = np.concatenate((charmat_low, charmat_high))
labels = np.concatenate((np.zeros(charmat_low.shape[0]), np.ones(charmat_high.shape[0])))
joblib.dump({"emotions": charmat, "labels": labels}, "combined_characteristics.gz")
