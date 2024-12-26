import pytest

import numpy as np

from boruta_shap_min.borutashap import BorutaShap, load_data


@pytest.mark.parametrize("method,sample", [('shap', False), ('gini', False), ('shap', True)])
def test_fit(method, sample):
    """
    Test all default values
    """
    X, y = load_data(data_type='classification')

    feature_selector = BorutaShap(importance_measure=method)
    feature_selector.fit(X=X, y=y, n_trials=5, random_state=0, train_or_test='train', sample=sample)

    assert feature_selector.n_trials == 5
    assert all(
        [(a == b).all() for a, b in
            zip(feature_selector.accepted_columns, [np.array([], dtype=object)] * 5)]
    )
