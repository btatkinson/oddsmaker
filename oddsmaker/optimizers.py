

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



class MasseyOptimizer:
    def __init__(self, decay_type, protag_col='team', antag_col='opponent', stat_col='team_sq_score', meta_cols=['location'], min_protag_games=5):
        self.decay_type = decay_type
        self.protag_col = protag_col
        self.antag_col = antag_col
        self.stat_col = stat_col
        self.meta_cols = meta_cols
        self.min_protag_games = min_protag_games

        if decay_type not in ['time', 'games', 'both']:
            raise ValueError("decay_type must be 'time', 'games', or 'both'")

    def load_data(self, data=None, path=None):
        if path:
            self.data = pd.read_csv(path)
        elif data is not None:
            self.data = data.copy()
        else:
            raise ValueError("Either data or path must be provided")
        self._preprocess_data()

    def _preprocess_data(self):
        required_columns = [self.protag_col, self.antag_col, self.stat_col, 'date']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

        self.protags = sorted(self.data[self.protag_col].unique())
        self.antags = sorted(self.data[self.antag_col].unique())

        protag_map = {p: i for i, p in enumerate(self.protags)}
        antag_map = {a: i for i, a in enumerate(self.antags)}

        self.data['protag_idx'] = self.data[self.protag_col].map(protag_map)
        self.data['antag_idx'] = self.data[self.antag_col].map(antag_map)

        if len(self.data) <= 200:
            raise ValueError("Not enough data to optimize (minimum 200 rows)")

    def _initialize_X(self, df, protags, antags):
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        num_protags = len(protags)
        num_antags = len(antags)
        
        protag_map = {p: i for i, p in enumerate(protags)}
        antag_map = {a: i for i, a in enumerate(antags)}
        
        df.loc[:, 'protag_idx'] = df[self.protag_col].map(protag_map).fillna(-1).astype(int)
        df.loc[:, 'antag_idx'] = df[self.antag_col].map(antag_map).fillna(-1).astype(int)
        
        X = sparse.lil_matrix((len(df), num_protags + num_antags + len(self.meta_cols)))
        valid_rows = (df['protag_idx'] != -1) & (df['antag_idx'] != -1)
        X[valid_rows, df.loc[valid_rows, 'protag_idx']] = 1
        X[valid_rows, df.loc[valid_rows, 'antag_idx'] + num_protags] = 1
        
        for i, col in enumerate(self.meta_cols):
            X[:, -(i+1)] = df[col].values.reshape(-1, 1)
        
        return sparse.csr_matrix(X), df[valid_rows]

    def _calculate_weights(self, train_data, test_date, halflife):
        decay = np.exp(-np.log(2) / halflife)
        time_diff = (test_date - train_data['date']).dt.total_seconds() / (24 * 3600)
        weights = decay ** time_diff
        return weights.values

    def _fit_model(self, X_train, y_train, weights, l2):
        W = sparse.diags(weights)
        q = (X_train.T @ W @ X_train).toarray()
        q += l2 * np.eye(q.shape[0]) * np.trace(q) / q.shape[0]
        f = X_train.T @ W @ y_train
        return solve(q, f, assume_a='pos')

    def _predict_and_evaluate(self, X_test, y_test, coeffs, num_protags, num_antags):
        offense_ratings = coeffs[:num_protags]
        defense_ratings = coeffs[num_protags:num_protags+num_antags]
        
        X_test_ratings = np.column_stack([
            offense_ratings[X_test[:, :num_protags].nonzero()[1]],
            defense_ratings[X_test[:, num_protags:num_protags+num_antags].nonzero()[1] - num_protags]
        ])
        
        linear_model = LinearRegression()
        predictions = cross_val_predict(linear_model, X_test_ratings, y_test, cv=5)
        mse = np.mean((y_test - predictions) ** 2)
        return mse

    def optimize(self, init_points=10, n_iter=30, num_test_dates=20, num_future_days=60, max_lookback=365*3, halflife_bounds=(10, 800), l2_bounds=(1e-9, 10)):
        unique_dates = sorted(self.data['date'].unique())[10:]
        test_dates = np.random.choice(unique_dates, size=num_test_dates, replace=False)

        def objective(halflife, l2):
            total_mse = 0
            for test_date in test_dates:
                train_data = self.data[(self.data['date'] >= test_date - pd.Timedelta(days=max_lookback)) & (self.data['date'] < test_date)].copy()
                test_data = self.data[(self.data['date'] >= test_date) & (self.data['date'] <= test_date + pd.Timedelta(days=num_future_days))].copy()

                if len(train_data) < 50 or len(test_data) < 50:
                    continue

                X_train, train_data = self._initialize_X(train_data, self.protags, self.antags)
                X_test, test_data = self._initialize_X(test_data, self.protags, self.antags)

                weights = self._calculate_weights(train_data, test_date, halflife)
                coeffs = self._fit_model(X_train, train_data[self.stat_col].values, weights, l2)

                mse = self._predict_and_evaluate(X_test, test_data[self.stat_col].values, coeffs, len(self.protags), len(self.antags))
                total_mse += mse

            return -total_mse / len(test_dates)

        optimizer = BayesianOptimization(f=objective, pbounds={'halflife': halflife_bounds, 'l2': l2_bounds}, random_state=17)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max['params']
        best_mse = -optimizer.max['target']
        return best_params['halflife'], best_params['l2'], best_mse

    def get_ratings_for_dates(self, dates, halflife, l2, max_lookback=365*2.1):
        offense_stats = []
        defense_stats = []

        for date in tqdm(dates):
            date = pd.to_datetime(date)
            train_data = self.data[(self.data['date'] >= date - pd.Timedelta(days=max_lookback)) & (self.data['date'] < date)].copy()
            
            if len(train_data) < 50:
                print(f"Minimum data threshold not met for date {date}")
                continue

            X_train, train_data = self._initialize_X(train_data, self.protags, self.antags)
            weights = self._calculate_weights(train_data, date, halflife)
            coeffs = self._fit_model(X_train, train_data[self.stat_col].values, weights, l2)

            num_protags = len(self.protags)
            num_antags = len(self.antags)
            
            offense_ratings = coeffs[:num_protags]
            defense_ratings = coeffs[num_protags:num_protags+num_antags]
            meta_ratings = coeffs[num_protags+num_antags:]

            offense_stats.append(pd.DataFrame({
                'protag': self.protags,
                self.stat_col: offense_ratings,
                'date': date
            }))

            defense_stats.append(pd.DataFrame({
                'antag': self.antags,
                self.stat_col: defense_ratings,
                'date': date
            }))

        return pd.concat(offense_stats), pd.concat(defense_stats)