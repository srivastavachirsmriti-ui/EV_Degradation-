"""Transparent domain vocabulary for EV text, independent of vehicle names.

These are expert-authored features, not mined from held-out labels. They
encode symptom concepts and question intent so five synthetic sentences per
class need not teach every synonym from scratch. They still require real-data
validation. Neither numbers, vehicle identity nor stored outputs set a label.
"""
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


CONCEPTS = {
    "thermal": r"\b(?:heat|heats|heating|hot|warm|overheat\w*|temperature|thermal|cooling)\b",
    "capacity": r"\b(?:capacity|degrad\w*|health|lost|loss|fallen|fade|fading|retention)\b",
    "range": r"\b(?:range|efficient|efficiency|energy per|highway|winter|route|trips?|driving conditions)\b",
    "charging": r"\b(?:charg\w*|charger\w*|plug\w*|recharg\w*)\b",
    "charge_failure": r"(?:slow\w*|longer|stops? charging|won.t charg\w*|not charg\w*|charg\w* speeds?|charging before|disconnect\w*)",
    "range_change": r"(?:less range|range estimate|range.*inconsistent|less efficient|more energy|charging stops|dashboard estimate)",
    "information": r"(?:looking for advice|comparing|want to know|how often|does anyone know|trying to understand|best.*routine|battery care|considered good)",
    "loss_specific": r"(?:lost.*(?:capacity|range)|usable.*(?:capacity|range)|capacity.*(?:drop|loss)|battery.*(?:fade|fading)|health.*(?:around|roughly))",
    "negation": r"\b(?:not|no|never|isn.t|hasn.t|without)\b",
}


class DomainFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        patterns = [re.compile(p) for p in CONCEPTS.values()]
        for text in X:
            values = [float(bool(p.search(str(text).lower()))) for p in patterns]
            thermal, capacity, distance, charge, failure, variation, information, loss, negation = values
            # Interactions retain context such as heat during charging and
            # general battery-health questions rather than actual capacity loss.
            rows.append(values + [thermal * charge, charge * failure, capacity * loss,
                                  distance * variation, capacity * information])
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array(list(CONCEPTS) + ["heat_while_charging", "charging_failure", "capacity_loss",
                                         "range_variation", "health_question"], dtype=object)
