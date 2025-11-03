"""ANOVA-style permutation tests for constrained ordination."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

from ..ordination.base import ConstrainedOrdinationResult
from .permtest import permutest as _permutest


def anova_cca(ordination_result: ConstrainedOrdinationResult,
              permutations: int = 999,
              *,
              random_state: Optional[Union[int, np.random.Generator]] = None) -> Dict[str, Union[pd.DataFrame, int, float, np.ndarray]]:
    """Permutation ANOVA for constrained ordination results.

    Parameters
    ----------
    ordination_result:
        Result object returned by :func:`nuee.rda` or :func:`nuee.cca`.
    permutations:
        Number of permutations used to approximate the null distribution.
    random_state:
        Optional seed/Generator for reproducible permutations.

    Returns
    -------
    dict
        Dictionary containing the ANOVA table and permutation metadata.
    """

    perm_result = _permutest(ordination_result, permutations=permutations, random_state=random_state)
    table = perm_result.get("tab")
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)

    return {
        "tab": table,
        "permutations": perm_result.get("permutations", permutations),
        "f_observed": perm_result.get("f_observed"),
        "f_permutations": perm_result.get("f_permutations")
    }
