from abc import ABC, abstractmethod


class FBRecommender(ABC):
    def __init__(self):
        assert hasattr(self, 'recommendations')
        assert hasattr(self, 'sim_scores_sorted')
        assert hasattr(self, 'sim_scores_argsorted')

    @abstractmethod
    def populate_recommender(self):
        raise NotImplementedError()
