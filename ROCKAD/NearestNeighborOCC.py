"""
Implementation of the Nearest Neighbor Method for one-class classification,
by Tax in [1].

[1] Tax, D.: One-class classification: Concept-learning in the absence of 
    counter-examples. PhD thesis, Delft University of Technology (2001)
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import distance_metrics


class NearestNeighborOCC():
    
    def __init__(self, dist="euclidean"):
        self.scores_train = None
        self.dist = None
        
        metrics = distance_metrics()
        
        if type(dist) is str and dist in metrics.keys():
            self.dist = metrics[dist]
        elif dist in metrics.values():
            self.dist = dist
        elif False:
            # TODO: allow time series distance measures such as DTW or Matrix Profile
            pass
        else:
            raise Exception("Distance metric not supported.")
    
    
    def fit(self, scores_train):
        _scores_train = scores_train
        
        if type(_scores_train) is not np.array:
            _scores_train = np.array(scores_train.copy())
        
        if len(_scores_train.shape) == 1:
            _scores_train = _scores_train.reshape(-1, 1)
        
        self.scores_train = _scores_train
        
        return self
    
        
    def predict(self, scores_test):
        """
        Per definition (see [1]): 0 indicates an anomaly, 1 indicates normal.
        Here : -1 indicates an anomaly, 1 indicates normal. 
        """
        
        predictions = []
        for score in scores_test:
            predictions.append(self.predict_score(score))
        return np.array(predictions)
    
    
    def predict_score(self, anomaly_score):
        prediction = None
        
        anomaly_score_arr = np.array([anomaly_score for i in range(len(self.scores_train))])
        
        _scores_train = self.scores_train.copy().reshape(-1, 1)
        anomaly_score_arr = anomaly_score_arr.reshape(-1, 1)
        nearest_neighbor_idx = np.argmin(self.dist(anomaly_score_arr, _scores_train))
        
        _scores_train = np.delete(_scores_train, nearest_neighbor_idx).reshape(-1, 1)
        
        nearest_neighbor_score = self.scores_train[nearest_neighbor_idx]
        neares_neighbot_score_arr = np.array([nearest_neighbor_score for i in range(len(_scores_train))])
        nearest_neighbor_score_arr = neares_neighbot_score_arr.reshape(-1, 1)
        
        nearest_nearest_neighbor_idx = np.argmin(self.dist(nearest_neighbor_score_arr, _scores_train))
        nearest_nearest_neighbor_score = _scores_train[nearest_nearest_neighbor_idx][0]
        
        prediction = self.indicator_function(
            anomaly_score, nearest_neighbor_score, nearest_nearest_neighbor_score)
        
        return prediction
    
    
    def indicator_function(self, z_score, nearest_score, nearest_of_nearest_score):
        
        # make it an array and reshape it to calculate the distance
        z_score_arr = np.array(z_score).reshape(1, -1)
        nearest_score_arr = np.array(nearest_score).reshape(1, -1)
        nearest_of_nearest_score_arr = np.array(nearest_of_nearest_score).reshape(1, -1)
        
        numerator = self.dist(z_score_arr, nearest_score_arr)
        denominator = self.dist(nearest_score_arr, nearest_of_nearest_score_arr)
        
        # error handling for corner cases
        if numerator == 0:
            return 1
        elif denominator == 0:
            return -1
        else:
            return 1 if (numerator/denominator) <= 1 else -1