
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from sktime.transformations.panel.rocket import Rocket

class NN:
    
    def __init__(self, 
            n_neighbors = 5, 
            n_jobs = 1,
            dist = 'euclidean',
            random_state=42, 
        ) -> None:
        
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.dist = dist
        self.random_state = random_state


    def fit(self, X):
        self.nn = NearestNeighbors(
            n_neighbors = self.n_neighbors,
            n_jobs = self.n_jobs,
            metric = self.dist,
            algorithm = 'ball_tree',
            )
        
        self.nn.fit(X)


    def predict_proba(self, X, y=None):
        scores = self.nn.kneighbors(X)
        scores = scores[0].mean(axis=1).reshape(-1,1)
        
        return scores
    


class ROCKAD():
    
    def __init__(self,
            n_estimators=10,
            n_kernels = 100,
            n_neighbors = 5,
            n_jobs = 1,
            power_transform = True,
            random_state = 42,
        ) -> None:
        self.random_state = random_state
        self.power_transform = power_transform
        
        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.n_inf_cols = []
        
        self.estimator = NN
        self.rocket_transformer = Rocket(num_kernels = self.n_kernels, n_jobs = self.n_jobs, random_state = self.random_state)
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(standardize = False)


    def init(self, X):
        
        # Fit Rocket & Transform into rocket feature space
        Xt = self.rocket_transformer.fit_transform(X)

        self.Xtp = None # X: values, t: (rocket) transformed, p: power transformed
        
        if self.power_transform is True:
            
            Xtp = self.power_transformer.fit_transform(Xt)
            
            self.Xtp = pd.DataFrame(Xtp)
            
        else:
            self.Xtp = pd.DataFrame(Xt)


    def fit_estimators(self):
        
        Xtp_scaled = None
        
        if self.power_transform is True:
            # Check for infinite columns and get indices
            self._check_inf_values(self.Xtp)
            
            # Remove infinite columns
            self.Xtp = self.Xtp[self.Xtp.columns[~self.Xtp.columns.isin(self.n_inf_cols)]]
            
            # Fit Scaler
            Xtp_scaled = self.scaler.fit_transform(self.Xtp)
            
            Xtp_scaled = pd.DataFrame(Xtp_scaled, columns = self.Xtp.columns)
            
            self._check_inf_values(Xtp_scaled)
            
            Xtp_scaled = Xtp_scaled.astype(np.float32).to_numpy()
            
        else:
            Xtp_scaled = self.Xtp.astype(np.float32).to_numpy()
            
        
        self.list_baggers = []
        
        for idx_estimator in range(self.n_estimators):
            # Initialize estimator
            estimator = self.estimator(
                n_neighbors = self.n_neighbors,
                n_jobs = self.n_jobs,
            )
            
            # Bootstrap Aggregation
            Xtp_scaled_sample = resample(
                Xtp_scaled,
                replace = True,
                n_samples = None,
                random_state = self.random_state + idx_estimator,
                stratify = None,
            )

            # Fit estimator and append to estimator list
            estimator.fit(Xtp_scaled_sample)
            self.list_baggers.append(estimator)


    def fit(self, X):
        self.init(X)
        self.fit_estimators()
        
        return self
    
    
    def predict_proba(self, X):
        y_scores = np.zeros((len(X), self.n_estimators))
        
        # Transform into rocket feature space
        Xt = self.rocket_transformer.transform(X)
        
        Xtp_scaled = None
        
        if self.power_transform == True:
            # Power Transform using yeo-johnson
            Xtp = self.power_transformer.transform(Xt)
            Xtp = pd.DataFrame(Xtp)
            
            # Check for infinite columns and remove them
            self._check_inf_values(Xtp)
            Xtp = Xtp[Xtp.columns[~Xtp.columns.isin(self.n_inf_cols)]]
            Xtp_temp = Xtp.copy()
            
            # Scale the data
            Xtp_scaled = self.scaler.transform(Xtp_temp)
            Xtp_scaled = pd.DataFrame(Xtp_scaled, columns = Xtp_temp.columns)
            
            # Check for infinite columns and remove them
            self._check_inf_values(Xtp_scaled)
            Xtp_scaled = Xtp_scaled[Xtp_scaled.columns[~Xtp_scaled.columns.isin(self.n_inf_cols)]]
            Xtp_scaled = Xtp_scaled.astype(np.float32).to_numpy() 
        
        else:
            Xtp_scaled = Xt.astype(np.float32)
        
        
        for idx, bagger in enumerate(self.list_baggers):
            # Get scores from each estimator
            scores = bagger.predict_proba(Xtp_scaled).squeeze()
            
            y_scores[:, idx] = scores
            
        # Average the scores to get the final score for each time series
        y_scores = y_scores.mean(axis=1)
        
        return y_scores
    
    
    def _check_inf_values(self, X):
        if np.isinf(X[X.columns[~X.columns.isin(self.n_inf_cols)]]).any(axis=0).any() : 
            self.n_inf_cols.extend(X.columns.to_series()[np.isinf(X).any()])
            self.fit_estimators()
            return True
    
    