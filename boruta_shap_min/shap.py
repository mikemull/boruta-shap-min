
import numpy as np
import shap


class ShapExplainer:

    def __init__(self, model, classification):
        self.model = model
        self.classification = classification

    def explain(self, x_boruta):
        """
        The shap package has numerous variants of explainers which use different assumptions depending on the model
        type this function allows the user to choose explainer

        Returns:
            shap values

        Raise
        ----------
            ValueError:
                if no model type has been specified tree as default
        """
        explainer = shap.TreeExplainer(
            self.model, feature_perturbation="tree_path_dependent", approximate=True
        )

        if self.classification:
            # for some reason shap returns values wraped in a list of length 1
            shap_vals = np.array(explainer.shap_values(x_boruta))
            if isinstance(shap_vals, list):
                class_inds = range(len(shap_vals))
                shap_imp = np.zeros(shap_vals[0].shape[1])
                for i, ind in enumerate(class_inds):
                    shap_imp += np.abs(shap_vals[ind]).mean(0)
                shap_vals /= len(shap_vals)

            elif len(shap_vals.shape) == 3:
                shap_vals = np.abs(shap_vals).sum(axis=0)
                shap_vals = shap_vals.mean(axis=1)

            else:
                shap_vals = np.abs(shap_vals).mean(0)

        else:
            shap_vals = explainer.shap_values(x_boruta)
            shap_vals = np.abs(shap_vals).mean(0)

        return shap_vals
