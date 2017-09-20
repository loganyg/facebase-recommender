# July 31, 2017
# Logan Young
#
# Implementation of alternating least squares method of single value
# decomposition collaborative filtering. Model from Yifan Hu et al.
# "Collaborative Filtering for Implicit Feedback Datasets".

import pandas as pd
import numpy as np
from numpy.linalg import inv


class SVDUIRecommender:

    def __init__(self, datasource, **kwargs):
        print("Using SVD User->Item Recommender")
        self.factor_num = kwargs.get('factor_num', 10)
        self.reg_factor = kwargs.get('reg_factor', 10)
        use_weights = kwargs.get('use_weights', True)
        # The datasource can be a tuple, pandas dataframe, or filename.
        if type(datasource) == tuple:
            # If the data is a tuple it should be a threeple of:
            # (ratings matrix, user names, item names)
            self.items = datasource[2]
            self.users = datasource[1]
            self.data = datasource[0]
        else:
            # Load data as a pandas DataFrame first to extract index
            # and column names.
            if type(datasource) == str:
                if datasource.endswith(".csv"):
                    pd_data = pd.read_csv(datasource, index_col=0)
                elif datasource.endswith(".pkl"):
                    pd_data = pd.read_pickle(datasource)
            elif type(datasource) == pd.DataFrame:
                pd_data = datasource
            # Stores items and users for convenience/readability.
            self.items = pd_data.columns
            self.users = pd_data.index
            # Convert the pandas data to a numpy array.
            self.data = pd_data.as_matrix()
        # Initialize the user and item factors to random values in [0,1].
        self.u_factors = np.random.rand(len(self.users), self.factor_num)
        self.i_factors = np.random.rand(len(self.items), self.factor_num)
        # Create a diagonal matrix filled with the regularization factor
        # for efficient calculation.
        self.reg_id = self.reg_factor * np.identity(self.factor_num)
        # Initialize empty similarity score matrices.
        self.sim_scores = np.empty((len(self.users), len(self.items)))
        self.sim_scores_argsorted = np.empty(
            (len(self.users), len(self.items))
        )
        self.sim_scores_sorted = np.empty((len(self.users), len(self.items)))
        # Initialize an empty pandas dataframe for user recommendations
        self.ui_recs = pd.DataFrame(
            index=self.users,
            columns=range(0, len(self.items))
        )
        # Create a preferences matrix indicating whether or not a user has
        # consumed the item.
        self.preferences = np.copy(self.data)
        self.preferences[self.preferences > 0] = 1
        # From Pan et al., assigns a static confidence weight to missing
        # preferences.
        if use_weights:
            weight_type = kwargs.get('weight_type', 'scale')
            weight_param = kwargs.get('weight_param', 30)
            # Assigns a confidence weight of 1 to positive preferences
            self.weights = np.copy(self.data).astype(float)
            # From Pan et al., assigns a static confidence weight
            # to missing preferences.
            if weight_type == 'uniform':
                self.weights[self.weights != 0] = 1
                self.weights[self.weights == 0] = weight_param
            # From Pan et al., assigns a confidence weight between 0 and 1
            # based n the number of preferences a user has.
            elif weight_type == 'user_oriented':
                self.weights[self.weights != 0] = 1
                hyperparam = 1 / float(max(np.sum(self.data, axis=1)))
                print(hyperparam)
                for u in range(0, len(self.users)):
                    c_u = sum(self.data[u, :])
                    row = self.weights[u]
                    row[row == 0] = hyperparam * c_u
            # From Pan et al., assigns a confidence weight between 0 and 1
            # based on the number of preferences an item has.
            elif weight_type == 'item_oriented':
                self.weights[self.weights != 0] = 1
                hyperparam = 1 / float(len(self.users))
                for i in range(0, len(self.items)):
                    c_i = sum(self.data[:, i])
                    col = self.weights[:, i]
                    col[col == 0] = hyperparam * (len(self.users) - c_i)
            # From Hu et al., assigns a weight > 1 on a linear scale of the
            # user item rating.
            elif weight_type == 'scale':
                self.weights = (self.weights * weight_param) + 1
        self.use_weights = use_weights

    def compute_u_factors(self):
        '''Assumes the item factors are fixed,
        reducing the cost function to a quadratic,
        then solving for values of the user factors,
        partially minimizing the cost function.'''
        if not self.use_weights:
            factor_matrix = np.dot(
                inv(
                    np.dot(
                        self.i_factors.T, self.i_factors
                    ) + self.reg_id
                ), self.i_factors.T)
            for u in range(0, len(self.users)):
                self.u_factors[u, :] = np.dot(
                    factor_matrix, self.preferences[u, :]
                )
        else:
            for u in range(0, len(self.users)):
                conf_mult_matrix = np.tile(
                    self.weights[u, :], (self.factor_num, 1))
                self.u_factors[u, :] = np.dot(
                    np.dot(
                        inv(
                            np.dot(
                                self.i_factors.T * conf_mult_matrix,
                                self.i_factors) + self.reg_id
                        ),
                        self.i_factors.T) * conf_mult_matrix,
                    self.preferences[u, :])

    def compute_i_factors(self):
        '''Assumes the user factors are fixed, reducing the cost function
        to a quadratic, then solving for values of the item factors,
        partially minimizing the cost function.'''
        if not self.use_weights:
            factor_matrix = np.dot(
                inv(
                    np.dot(
                        self.u_factors.T, self.u_factors
                    ) + self.reg_id
                ), self.u_factors.T
            )
            for i in range(0, len(self.items)):
                self.i_factors[i, :] = np.dot(
                    factor_matrix, self.preferences[:, i]
                )
        else:
            for i in range(0, len(self.items)):
                conf_mult_matrix = np.tile(
                    self.weights[:, i], (self.factor_num, 1))
                self.i_factors[i, :] = np.dot(
                    np.dot(
                        inv(
                            np.dot(
                                self.u_factors.T * conf_mult_matrix,
                                self.u_factors
                            ) + self.reg_id
                        ),
                        self.u_factors.T) * conf_mult_matrix,
                    self.preferences[:, i])

    def cost(self):
        '''Performs a naive calculation of the cost function
        for the current item and user factor matrices, then returns
        the calculated value.'''
        cost = 0
        for u in range(0, len(self.users)):
            for i in range(0, len(self.items)):
                cost += self.weights[u, i] * (
                    (
                        float(self.data[u, i]) - np.dot(
                            self.i_factors[i], self.u_factors[u]
                        )
                    )**2
                )
        sum_term = 0
        for u in range(0, len(self.users)):
            sum_term += np.linalg.norm(self.u_factors[u])**2
        for i in range(0, len(self.items)):
            sum_term += np.linalg.norm(self.i_factors[i])**2
        cost += self.reg_factor * sum_term
        return cost

    def populate_recommender(self, **kwargs):
        '''Alternates the least squares calculations for user and
        item factor matrices for a specified number
        of iterations, then calculates the predicted
        user-item similarities based on the factor matrices.
        Finally, finds a certain top items for each
        users and arranges them into a DataFrame.'''
        iterations = kwargs.get('iterations', 10)
        show_cost = kwargs.get('show_cost', False)
        heuristic_threshold = kwargs.get('heuristic_threshold', 0)
        print("Heuristic Threshold is actually %s" % heuristic_threshold)
        for _ in range(0, iterations):
            self.compute_i_factors()
            self.compute_u_factors()
            if show_cost:
                print(self.cost())
        user_sums = np.sum(self.preferences, axis=1)
        item_pop = np.sum(self.preferences, axis=0)
        for u in range(0, len(self.users)):
            for i in range(0, len(self.items)):
                if self.preferences[u, i] == 1:
                    self.sim_scores[u, i] = 0
                else:
                    if user_sums[u] <= heuristic_threshold:
                        self.sim_scores[u, i] = item_pop[i]
                    else:
                        self.sim_scores[u, i] = np.dot(
                            self.u_factors[u, :], self.i_factors[i, :])
        self.sim_scores_argsorted = np.argsort(self.sim_scores, axis=1)
        self.sim_scores_sorted = np.sort(self.sim_scores, axis=1)
        for u in range(0, len(self.users)):
            self.ui_recs.ix[self.users[u], :] = self.items[
                np.flip(self.sim_scores_argsorted[u], 0)
            ]
