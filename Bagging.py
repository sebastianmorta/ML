from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BaggingEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, hard_voting=True):
        print('h')
