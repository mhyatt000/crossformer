from __future__ import annotations

import pytest


def test_third_party_tf():
    pass


@pytest.mark.skip
def test_third_party_xgym():
    pass


@pytest.mark.skip
def test_third_party_xclients():
    pass


@pytest.mark.skip
def test_third_party_dlimp():
    pass


def test_import_top_level():
    """Test that top-level crossformer package imports without error."""
    import crossformer

    assert hasattr(crossformer, "BASE")
    assert hasattr(crossformer, "ROOT")


def test_head():
    """Test importing key classes and utilities from main submodules."""
    from crossformer.model.components.heads import (
        ActionHead,
        ContinuousActionHead,
        DiffusionActionHead,
        FlowMatchingActionHead,
    )

    assert ActionHead is not None
    assert ContinuousActionHead is not None
    assert DiffusionActionHead is not None
    assert FlowMatchingActionHead is not None


def test_cn():
    pass
    # import crossformer.cn.wab
    # import crossformer.cn.model # fail

    # assert crossformer.cn is not None
    # assert CN is not None
    # assert Train is not None
    # assert Dataset is not None
    # assert Experiment is not None


def test_submodule_data():
    """Test that the data submodule can be imported and contains expected items."""
    import crossformer.data

    assert crossformer.data is not None


def test_submodules_import():
    """Test that submodules can be imported directly."""
    import crossformer.model
    import crossformer.model.components
    import crossformer.model.components.heads
    import crossformer.utils

    assert crossformer.model is not None
    assert crossformer.model.components is not None
    assert crossformer.model.components.heads is not None
    assert crossformer.utils is not None


def test_heads_submodule_imports():
    """Test importing specific items from nested submodules."""
    from crossformer.model.components.heads import (
        ActionHead,
        AdjFlowHead,
        continuous_loss,
        ContinuousActionHead,
        DiffusionActionHead,
        FlowMatchingActionHead,
        L1ActionHead,
        masked_mean,
        MSEActionHead,
        sample_tau,
    )

    assert callable(ActionHead)
    assert callable(ContinuousActionHead)
    assert callable(L1ActionHead)
    assert callable(MSEActionHead)
    assert callable(DiffusionActionHead)
    assert callable(FlowMatchingActionHead)
    assert callable(AdjFlowHead)
    assert callable(masked_mean)
    assert callable(continuous_loss)
    assert callable(sample_tau)


pytestmark = pytest.mark.unit
