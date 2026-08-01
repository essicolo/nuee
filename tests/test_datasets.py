from importlib import resources

import pandas as pd
import pytest

import nuee


EXPECTED_SHAPES = {
    "varespec": (24, 44),
    "varechem": (24, 14),
    "dune": (20, 30),
    "dune_env": (20, 5),
    "dune_taxon": (30, 5),
    "BCI": (50, 225),
    "BCI_env": (50, 9),
    "mite": (70, 35),
    "mite_env": (70, 5),
    "mite_pcnm": (70, 22),
    "mite_xy": (70, 2),
    "pyrifos": (132, 178),
    "sipoo": (18, 50),
    "sipoo_map": (18, 3),
}


def test_every_bundled_csv_has_a_loader():
    """Regression: seven CSVs shipped in every install with no way to reach
    them from the public API."""
    shipped = {
        path.name for path in resources.files("nuee.data").iterdir()
        if path.name.endswith(".csv")
    }
    assert len(shipped) == len(nuee.datasets.__all__)


@pytest.mark.parametrize("name,shape", sorted(EXPECTED_SHAPES.items()))
def test_dataset_loads_with_expected_shape(name, shape):
    frame = getattr(nuee.datasets, name)()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == shape
    # The first CSV column is the site/row label, not data.
    assert frame.index.name is None or frame.index.name != frame.columns[0]


def test_all_exports_are_importable():
    for name in nuee.datasets.__all__:
        assert callable(getattr(nuee.datasets, name))
