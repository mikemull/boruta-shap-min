import pytest

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from boruta_shap_min.borutashap import BorutaShap, load_data


@pytest.mark.parametrize("method,sample", [('shap', False), ('gini', False), ('shap', True)])
def test_fit(method, sample):
    """
    Test all default values
    """
    X, y = load_data(data_type='classification')

    n_trials = 5
    feature_selector = BorutaShap(importance_measure=method)
    feature_selector.fit(X=X, y=y, n_trials=n_trials, random_state=0, train_or_test='train', sample=sample)

    assert feature_selector.n_trials == 5
    assert all(
        [(a == b).all() for a, b in
            zip(feature_selector.accepted_columns, [np.array([], dtype=object)] * n_trials)]
    )


def test_fit_regression():
    """
    Test all default values
    """
    X, y = load_data(data_type='regression')

    feature_selector = BorutaShap(
        model=DecisionTreeRegressor(),
        importance_measure='shap',
        classification=False
    )
    n_trials = 20
    feature_selector.fit(X=X, y=y, n_trials=n_trials, random_state=0, train_or_test='train')

    assert feature_selector.n_trials == n_trials
    assert set(feature_selector.accepted) == set(['bmi', 's5'])
