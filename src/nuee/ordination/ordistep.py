"""
Stepwise model selection for ordination.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any

from .base import OrdinationResult
from .rda import rda


def _as_frame(env: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """Return *env* as a DataFrame with usable column labels."""
    if isinstance(env, pd.DataFrame):
        return env
    array = np.asarray(env, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    return pd.DataFrame(array, columns=[f"X{i + 1}" for i in range(array.shape[1])])


def _term_pvalue(response,
                 term: pd.DataFrame,
                 conditioning: Optional[pd.DataFrame],
                 permutations: int,
                 random_state: Optional[int]) -> float:
    """Permutation p-value for one term, given the terms already retained.

    The term is tested as a constraint on top of the retained ones, which
    enter as the conditioning matrix of a partial RDA. That is the marginal
    test vegan's ordistep performs.
    """
    from ..permutation.permtest import permutest

    if conditioning is not None and conditioning.shape[1] > 0:
        model = rda(response, term, Z=conditioning)
    else:
        model = rda(response, term)

    table = permutest(model, permutations=permutations,
                      random_state=random_state)["tab"]
    return float(table.loc["Model", "Pr(>F)"])


def ordistep(ordination: OrdinationResult,
             env: Union[np.ndarray, pd.DataFrame],
             direction: str = "both",
             permutations: int = 199,
             p_enter: float = 0.05,
             p_remove: float = 0.1,
             max_steps: int = 50,
             random_state: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
    """
    Stepwise model selection for constrained ordination.

    Terms are added and/or dropped one at a time using permutation tests, in
    the manner of vegan's ``ordistep``. A candidate is tested as a constraint
    conditioned on the terms already retained, so each test is marginal.

    Parameters
    ----------
    ordination:
        A fitted constrained ordination (e.g. from :func:`nuee.rda`), used as
        the source of the response matrix.
    env:
        Candidate explanatory variables, one per column.
    direction:
        ``"forward"`` only adds terms, ``"backward"`` starts from the full set
        and only drops them, ``"both"`` alternates.
    permutations:
        Permutations per test.
    p_enter:
        A term is added when its p-value is at or below this.
    p_remove:
        A retained term is dropped when its p-value rises above this.
    max_steps:
        Safety bound on the number of add/drop rounds.
    random_state:
        Seed, for reproducible selection.

    Returns
    -------
    dict
        ``selected_variables`` (in the order chosen), ``p_values`` for those
        terms, the ``direction`` used, and the ``steps`` taken.

    Raises
    ------
    ValueError
        If *direction* is not one of the three accepted values, or the
        ordination carries no response matrix to select against.
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError(
            "direction must be 'forward', 'backward' or 'both', "
            f"got {direction!r}."
        )

    response = getattr(ordination, "_raw_response", None)
    if response is None:
        raise ValueError(
            "The ordination carries no response matrix; refit it with "
            "nuee.rda() before running ordistep()."
        )

    candidates = _as_frame(env)
    names = list(candidates.columns)

    selected: List[str] = list(names) if direction == "backward" else []
    p_values: Dict[str, float] = {}
    steps: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        changed = False

        # --- forward: add the most significant remaining term ---
        if direction in ("forward", "both"):
            remaining = [name for name in names if name not in selected]
            scores = {}
            for name in remaining:
                conditioning = candidates[selected] if selected else None
                scores[name] = _term_pvalue(response, candidates[[name]],
                                            conditioning, permutations,
                                            random_state)
            if scores:
                best = min(scores, key=scores.get)
                if scores[best] <= p_enter:
                    selected.append(best)
                    p_values[best] = scores[best]
                    steps.append({"action": "add", "term": best,
                                  "p_value": scores[best]})
                    changed = True

        # --- backward: drop the least significant retained term ---
        if direction in ("backward", "both") and selected:
            scores = {}
            for name in selected:
                others = [other for other in selected if other != name]
                conditioning = candidates[others] if others else None
                scores[name] = _term_pvalue(response, candidates[[name]],
                                            conditioning, permutations,
                                            random_state)
            worst = max(scores, key=scores.get)
            if scores[worst] > p_remove:
                selected.remove(worst)
                p_values.pop(worst, None)
                steps.append({"action": "drop", "term": worst,
                              "p_value": scores[worst]})
                changed = True
            else:
                p_values.update(scores)

        if not changed:
            break

    return {
        "selected_variables": selected,
        "p_values": {name: p_values[name] for name in selected if name in p_values},
        "direction": direction,
        "steps": steps,
        "permutations": permutations,
    }
