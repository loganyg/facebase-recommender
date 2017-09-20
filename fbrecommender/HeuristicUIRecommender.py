import pandas as pd
import numpy as np


class HeuristicUIRecommender:

    def __init__(self, datasource, **kwargs):
        print("Using Heuristic User->Item Recommender")
        if type(datasource) == tuple:
            # Possible to introduce data as a tuple of
            # (numpy array, user names, item names)
            self.items = datasource[2]
            self.users = datasource[1]
            self.data = datasource[0]
        else:
            # Load data as a pandas DataFrame first to extract index
            # and column names.
            if type(datasource) == str:
                if datasource.endswith(".csv"):
                    pd_data = pd.read_csv(datasource)
                elif datasource.endswith(".pkl"):
                    pd_data = pd.read_pickle(datasource)
            elif type(datasource) == pd.DataFrame:
                pd_data = datasource
            # Stores items and users for convenience/readability.
            self.items = pd_data.ix[:, 1:].columns
            self.users = pd_data.ix[:, 0]
            # Convert the pandas data to a numpy array.
            self.data = pd_data.ix[:, 1:].as_matrix()
        self.preferences = np.copy(self.data)
        self.preferences[self.preferences > 0] = 1

    def populate_recommender(self, **kwargs):
        heuristic = kwargs.get('heuristic', 'popularity')
        self.rec_matrix = np.empty((len(self.users), len(self.items)))
        if heuristic == 'popularity':
            self.sim_scores = np.tile(
                np.sum(self.preferences, axis=0),
                (len(self.users), 1)
            )
            for u in range(0, len(self.users)):
                for i in range(0, len(self.items)):
                    if self.preferences[u, i] == 1:
                        self.sim_scores[u, i] = 0
            self.sim_scores_sorted = np.sort(self.sim_scores, axis=1)
            self.sim_scores_argsorted = np.argsort(
                self.sim_scores,
                axis=1
            )
            for u in range(0, len(self.users)):
                self.u_recs.ix[self.users[u], :] = self.items[
                    self.sim_scores_argsorted[u, :]
                ]
