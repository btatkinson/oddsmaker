

import torch
import numpy as np
import pandas as pd

from datetime import datetime
from scipy.optimize import minimize

class Optimizer:
    def __init__(self):
        pass

    def optimize(self):
        raise NotImplementedError("Subclasses must implement the optimize method.")


class MasseyOptimizer(Optimizer):
    def __init__(self, decay_type, home_field=False):
        super().__init__()
        assert(decay_type in ['time', 'games', 'n_days','n_games', 'both'])
        self.decay_type = decay_type
        self.home_field = home_field
        self.data = None

    def load_data(self, data):
        self.data = data
        self.preprocess_data()

    def preprocess_data(self):
        # Convert date column to datetime if needed
        if isinstance(self.data['date'].iloc[0], str):
            self.data['date'] = pd.to_datetime(self.data['date'])

    def calculate_ratings(self, train_data, weights, l2):

        # Prepare input data
        if 'stat_1' in train_data.columns:
            X = torch.tensor(train_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
            y = torch.tensor(train_data['stat_1'].values - train_data['stat_2'].values, dtype=torch.float32)
        else:
            X = torch.tensor(train_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
            y = torch.tensor(train_data['stat'].values, dtype=torch.float32)

        # Solve the linear system with L2 regularization
        coefficients = torch.linalg.solve(X.T @ (X * weights.view(-1, 1)) + l2 * torch.eye(X.shape[1]), X.T @ (y * weights))

        return coefficients

    def calculate_time_weights(self, decay_factor, days_ago):
        return torch.exp(-decay_factor * days_ago)
    
    def time_optimize(self, num_test_dates=20, min_weight=0.01, num_future_days=60, decay_bounds=(0.001, 1.0), l2_bounds=(0.01, 100)):

        # Select random test dates
        unique_dates = self.data['date'].unique()
        test_dates = np.random.choice(unique_dates, size=num_test_dates, replace=False)

        def objective(params):
            decay_factor, l2 = params
            correlations = []

            for test_date in test_dates:
                # Filter data before the test date
                train_data = self.data[self.data['date'] < test_date]

                # Calculate days_ago for each game
                days_ago = (test_date - train_data['date']).dt.days
                weights = self.calculate_weights(decay_factor, torch.tensor(days_ago.values, dtype=torch.float32))

                # Filter games with weights greater than min_weight
                train_data = train_data[weights > min_weight]
                weights = weights[weights > min_weight]

                coefficients = self.calculate_ratings(train_data, weights, l2)

                # Filter future data and calculate predicted values
                future_data = self.data[(self.data['date'] >= test_date) & (self.data['date'] < test_date + pd.Timedelta(days=num_future_days))]
                if 'stat_1' in future_data.columns:
                    future_X = torch.tensor(future_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                    future_y_true = torch.tensor(future_data['stat_1'].values - future_data['stat_2'].values, dtype=torch.float32)
                else:
                    future_X = torch.tensor(future_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                    future_y_true = torch.tensor(future_data['stat'].values, dtype=torch.float32)

                future_y_pred = future_X @ coefficients

                correlation = np.corrcoef(future_y_true.numpy(), future_y_pred.numpy())[0, 1]
                correlations.append(correlation)

            avg_correlation = np.mean(correlations)
            return -avg_correlation  # Minimize the negative of the average correlation

        # Define the bounds for decay factor and L2 regularization
        bounds = [decay_bounds, l2_bounds]

        # Perform the optimization
        result = minimize(objective, x0=[0.1, 1], bounds=bounds)

        best_decay_factor, best_l2 = result.x
        best_correlation = -result.fun

        return best_decay_factor, best_l2, best_correlation

    def optimize(self, num_test_dates=20, min_weight=0.01, num_future_days=60):
        if self.decay_type != 'time':
            raise NotImplementedError("Only time decay is currently supported.")

        # Select random test dates
        unique_dates = self.data['date'].unique()
        test_dates = np.random.choice(unique_dates, size=num_test_dates, replace=False)

        best_decay_factor = None
        best_l2 = None
        best_correlation = -1

        for decay_factor in [0.01, 0.05, 0.1, 0.2, 0.5]:
            for l2 in [0.1, 1, 10, 100]:
                correlations = []

                for test_date in test_dates:
                    # Filter data before the test date
                    train_data = self.data[self.data['date'] < test_date]

                    # Calculate days_ago for each game
                    days_ago = (test_date - train_data['date']).dt.days
                    weights = self.calculate_weights(decay_factor, torch.tensor(days_ago.values, dtype=torch.float32))

                    # Filter games with weights greater than min_weight
                    train_data = train_data[weights > min_weight]
                    weights = weights[weights > min_weight]

                    # Prepare input data
                    if 'stat_1' in train_data.columns:
                        X = torch.tensor(train_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                        y = torch.tensor(train_data['stat_1'].values - train_data['stat_2'].values, dtype=torch.float32)
                    else:
                        X = torch.tensor(train_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                        y = torch.tensor(train_data['stat'].values, dtype=torch.float32)

                    # Solve the linear system with L2 regularization
                    coefficients = torch.linalg.solve(X.T @ (X * weights.view(-1, 1)) + l2 * torch.eye(X.shape[1]), X.T @ (y * weights))

                    # Filter future data and calculate predicted values
                    future_data = self.data[(self.data['date'] >= test_date) & (self.data['date'] < test_date + pd.Timedelta(days=num_future_days))]
                    if 'stat_1' in future_data.columns:
                        future_X = torch.tensor(future_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                        future_y_true = torch.tensor(future_data['stat_1'].values - future_data['stat_2'].values, dtype=torch.float32)
                    else:
                        future_X = torch.tensor(future_data[['team_1', 'team_2', 'meta_variable']].values, dtype=torch.float32)
                        future_y_true = torch.tensor(future_data['stat'].values, dtype=torch.float32)

                    future_y_pred = future_X @ coefficients

                    # Calculate correlation coefficient
                    correlation = np.corrcoef(future_y_true.numpy(), future_y_pred.numpy())[0, 1]
                    correlations.append(correlation)

                avg_correlation = np.mean(correlations)
                if avg_correlation > best_correlation:
                    best_decay_factor = decay_factor
                    best_l2 = l2
                    best_correlation = avg_correlation

        return best_decay_factor, best_l2, best_correlation