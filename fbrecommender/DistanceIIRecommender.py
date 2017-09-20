import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


class DistanceIIRecommender:

    def __init__(self, datasource, **kwargs):
        print("Using Distance Item->Item Recommender")
        self.factor_num = kwargs.get('factor_num', 10)
        self.reg_factor = kwargs.get('reg_factor', 10)
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
        self.ii_sims = np.empty((len(self.items), len(self.items)))
        self.ii_sims_argsorted = None
        self.ii_sims_sorted = None
        self.ii_recs = pd.DataFrame(
            index=self.items,
            columns=range(0, len(self.items))
        )

    def populate_recommender(self, dist_measure='cosine'):
        if dist_measure == "cosine":
            def calc_sim(x, y):
                return 1 - cosine(
                    self.data[:, x],
                    self.data[:, y]
                )
        for i in range(0, len(self.items)):
            for j in range(0, len(self.items)):
                if i == j:
                    self.ii_sims[i, j] = 0
                else:
                    self.ii_sims[i, j] = calc_sim(i, j)
        self.ii_sims_argsorted = np.flipud(np.argsort(self.ii_sims, axis=0))
        self.ii_sims_sorted = np.flipud(np.sort(self.ii_sims, axis=0))
        for i in range(0, len(self.items)):
            self.ii_recs.iloc[i, :] = self.items[
                self.ii_sims_argsorted[:, i]
            ]
