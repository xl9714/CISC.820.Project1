import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PiecewiseRegressor:
    def __init__(self, n_pieces=4, alpha=0.1, poly_degree=2, min_samples_leaf=10):
        self.n_pieces = n_pieces
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.min_samples_leaf = min_samples_leaf
        self.scaler = StandardScaler()
        self.models = {}
        self.tree = None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        self.tree = DecisionTreeRegressor(
            max_leaf_nodes=self.n_pieces,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_leaf * 2,
            max_depth=8,
            random_state=42
        )
        segments = self.tree.fit(X_scaled, y).apply(X_scaled)
        
        unique_segments = np.unique(segments)
        
        for segment_id in unique_segments:
            mask = segments == segment_id
            if np.sum(mask) < 5:
                continue
                
            X_segment = X_scaled[mask]
            y_segment = y[mask]
            
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
            X_poly = poly.fit_transform(X_segment)
            
            ridge = Ridge(alpha=self.alpha)
            ridge.fit(X_poly, y_segment)
            
            self.models[segment_id] = {
                'regressor': ridge,
                'poly': poly,
                'mean_y': np.mean(y_segment)
            }
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        segments = self.tree.apply(X_scaled)
        predictions = np.zeros(len(X))
        
        for i, segment_id in enumerate(segments):
            if segment_id in self.models:
                model_info = self.models[segment_id]
                X_point = X_scaled[i:i+1]
                X_poly = model_info['poly'].transform(X_point)
                predictions[i] = model_info['regressor'].predict(X_poly)[0]
            else:
                available_segments = list(self.models.keys())
                if available_segments:
                    closest_segment = min(available_segments, key=lambda x: abs(x - segment_id))
                    model_info = self.models[closest_segment]
                    predictions[i] = model_info['mean_y']
                else:
                    predictions[i] = 0
        
        return predictions

class PiecewiseEnsemble:
    def __init__(self, n_models=5, **regressor_params):
        self.n_models = n_models
        self.regressor_params = regressor_params
        self.models = []
        
    def fit(self, X, y):
        self.models = []
        for seed in range(self.n_models):
            np.random.seed(seed + 42)
            indices = np.random.choice(len(X), size=int(0.9 * len(X)), replace=False)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            model = PiecewiseRegressor(**self.regressor_params)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

def load_data():
    traindata = np.loadtxt('traindata.txt')
    X_train = traindata[:, :8]
    y_train = traindata[:, 8]
    X_test = np.loadtxt('testinputs.txt')
    return X_train, y_train, X_test

def evaluate_piecewise_model(X_train, y_train):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kfold.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = PiecewiseEnsemble(n_models=3)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        cv_scores.append(mean_squared_error(y_val, y_pred))
    
    return np.mean(cv_scores), np.std(cv_scores)

def main():
    X_train, y_train, X_test = load_data()
    
    print("Evaluating piecewise model with 10-fold CV...")
    cv_mean, cv_std = evaluate_piecewise_model(X_train, y_train)
    print(f"Piecewise model 10-fold CV: {cv_mean:.6f} ± {cv_std:.6f}")
    
    # Train the besdt final piecewise model
    print("\nTraining final piecewise model...")
    final_model = PiecewiseEnsemble(n_models=5)
    final_model.fit(X_train, y_train)
    
    train_pred = final_model.predict(X_train)
    test_pred = final_model.predict(X_test)
    
    train_error = mean_squared_error(y_train, train_pred)
    
    print(f"Piecewise model training error: {train_error:.6f}")
    print(f"Piecewise model estimated test error (10-fold CV): {cv_mean:.6f}")
    
    np.savetxt('predictions.txt', test_pred, fmt='%.6f')
    print("Predictions saved to 'predictions.txt'")
    
    return final_model, test_pred

if __name__ == "__main__":
    model, predictions = main()
