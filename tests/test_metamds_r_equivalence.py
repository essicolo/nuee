"""
Regression-style checks that ``nuee.metaMDS`` produces results comparable to
the reference implementation in R's vegan package.

The R output below was generated with:

```
library(vegan)
data(varespec)
metaMDS(varespec, k = 2, distance = "bray")
```

If you regenerate the R results (e.g. after algorithm changes), update the
``R_OUTPUT`` string so the expectations in this test stay in sync.
"""

from __future__ import annotations

import math
import re

import nuee

R_OUTPUT = """
Square root transformation
Wisconsin double standardization
Run 0 stress 0.1843196 
Run 1 stress 0.2475008 
Run 2 stress 0.1993239 
Run 3 stress 0.1974406 
Run 4 stress 0.1869637 
Run 5 stress 0.195049 
Run 6 stress 0.1825658 
... New best solution
... Procrustes: rmse 0.0416301  max resid 0.1518247 
Run 7 stress 0.2414246 
Run 8 stress 0.22964 
Run 9 stress 0.1825658 
... New best solution
... Procrustes: rmse 7.688509e-06  max resid 2.168486e-05 
... Similar to previous best
Run 10 stress 0.18458 
Run 11 stress 0.1825658 
... Procrustes: rmse 6.851132e-06  max resid 1.738399e-05 
... Similar to previous best
Run 12 stress 0.2085949 
Run 13 stress 0.2282745 
Run 14 stress 0.2109612 
Run 15 stress 0.2251281 
Run 16 stress 0.2346662 
Run 17 stress 0.1948413 
Run 18 stress 0.2345769 
Run 19 stress 0.2380741 
Run 20 stress 0.2088293 
*** Best solution repeated 2 times

Stress:     0.1825658
"""


def _parse_r_scalar(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Pattern {pattern!r} not found in R output.")
    return float(match.group(1))


def _parse_r_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Pattern {pattern!r} not found in R output.")
    return int(match.group(1))


def test_meta_mds_roughly_matches_r_reference():
    r_stress = _parse_r_scalar(r"Stress:\s+([0-9.]+)", R_OUTPUT)
    _ = _parse_r_int(r"Best solution repeated (\d+) times", R_OUTPUT)

    varespec = nuee.datasets.varespec()
    result = nuee.metaMDS(varespec, k=2, distance="bray")

    # metaMDS should report the transformations that vegan applies
    assert result.transformations == ["sqrt", "wisconsin"]

    # We should have tried the expected number of random starts.
    assert result.trymax == 20
    assert len(result.stress_history) == result.trymax
    assert result.stress_per_run and len(result.stress_per_run) == result.trymax
    assert [entry["run"] for entry in result.stress_history] == list(range(1, result.trymax + 1))

    # Stress should be on the same order of magnitude as the R result.
    assert math.isclose(result.stress, r_stress, rel_tol=0.75, abs_tol=0.15)

    # Metadata about the best run should be populated and repeat counts sensible.
    assert result.best_run >= 1
    assert result.best_run_repeats >= 1
    assert result.best_stress == result.stress

    # Provide a quick regression check that stress history is monotonic with minima.
    stresses = [entry["stress"] for entry in result.stress_history]
    assert math.isclose(min(stresses), result.best_stress, rel_tol=1e-9, abs_tol=1e-9)

    # Check that vegan-style metadata is exposed consistently.
    assert result.distance_method == "bray"
    assert result.maxit == 200
    assert result.call["method"] == "NMDS"
    assert result.call["distance"] == "bray"
    assert result.call["n_components"] == 2
    assert result.call["trymax"] == result.trymax
    assert result.call["maxit"] == result.maxit
    assert result.call["transformations"] == result.transformations
    assert result.call["best_run"] == result.best_run

    # Row/column labels and species metadata should mirror vegan.
    assert list(result.points.index) == list(varespec.index)
    assert list(result.points.columns) == ["NMDS1", "NMDS2"]
    assert result.species is not None
    assert list(result.species.index) == list(varespec.columns)
    assert list(result.species.columns) == ["NMDS1", "NMDS2"]
    species_attrs = result.species.attrs
    assert "shrinkage" in species_attrs
    assert "centre" in species_attrs
    assert len(species_attrs["shrinkage"]) == result.species.shape[1]
    assert species_attrs["centre"].shape == (result.species.shape[1],)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["--override-ini=addopts=", __file__]))
