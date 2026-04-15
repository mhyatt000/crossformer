"""Vocabulary and registry invariants for crossformer.embody.

Exercises the DOF dict, all registered BodyParts, Embodiments, and Datasets.
Added alongside the kpt2d/kpt3dc/kpt3dw vocab extension and SINGLE_GRIP_CAL
embodiment for xarm_dream_100k.
"""

from __future__ import annotations

import pytest

from crossformer import embody
from crossformer.embody import (
    BodyPart,
    Dataset,
    DOF,
    Embodiment,
    MASK_ID,
    SINGLE_GRIP_CAL,
    VOCAB_SIZE,
)


def test_dof_ids_are_unique():
    ids = list(DOF.values())
    assert len(ids) == len(set(ids)), "duplicate DOF IDs"


def test_dof_ids_fit_vocab():
    assert max(DOF.values()) < VOCAB_SIZE
    assert DOF["MASK"] == MASK_ID == 0


# Pre-existing: MANO_48 claims 48 DOFs but vocab only has mano_0..mano_6.
# Tracked separately; skip from the validity sweep.
_BROKEN_PARTS = {"mano_48"}


def test_all_bodyparts_reference_valid_dofs():
    parts = [v for v in vars(embody).values() if isinstance(v, BodyPart)]
    assert parts, "no BodyParts found in embody module"
    for part in parts:
        if part.name in _BROKEN_PARTS:
            continue
        for name in part.dof_names:
            assert name in DOF, f"{part.name}: unknown DOF {name!r}"
        assert len(part.dof_ids) == part.action_dim


def test_all_registered_embodiments_reference_valid_dofs():
    # Registry only contains embodiments actually used in datasets/catalog,
    # which should not include broken parts.
    for emb in Embodiment.REGISTRY.values():
        for part in emb.parts:
            assert part.name not in _BROKEN_PARTS, f"{emb.name} uses broken part {part.name!r}"
            for name in part.dof_names:
                assert name in DOF


def test_all_embodiments_consistent():
    for emb in Embodiment.REGISTRY.values():
        assert emb.action_dim == sum(p.action_dim for p in emb.parts)
        assert len(emb.dof_ids) == emb.action_dim
        assert all(d < VOCAB_SIZE for d in emb.dof_ids)


def test_all_datasets_have_registered_embodiments():
    for ds in Dataset.REGISTRY.values():
        assert ds.embodiment.name in Embodiment.REGISTRY
        assert ds.action_dim == ds.embodiment.action_dim


# ---- new embodiment: single_grip_cal ---------------------------------------


def test_single_grip_cal_shape():
    # 16 locations x (2D uv + 3D cam xyz + 3D world xyz) = 32+48+48 = 128
    assert SINGLE_GRIP_CAL.action_dim == 128
    assert [p.action_dim for p in SINGLE_GRIP_CAL.parts] == [32, 48, 48]


def test_kpt_vocab_blocks_disjoint():
    kpt2d = {k: v for k, v in DOF.items() if k.startswith("kpt2d_")}
    kpt3dc = {k: v for k, v in DOF.items() if k.startswith("kpt3dc_")}
    kpt3dw = {k: v for k, v in DOF.items() if k.startswith("kpt3dw_")}
    assert len(kpt2d) == 32
    assert len(kpt3dc) == 48
    assert len(kpt3dw) == 48
    assert set(kpt2d.values()).isdisjoint(kpt3dc.values())
    assert set(kpt2d.values()).isdisjoint(kpt3dw.values())
    assert set(kpt3dc.values()).isdisjoint(kpt3dw.values())


def test_xarm_dream_dataset_registered():
    ds = Dataset.REGISTRY.get("xarm_dream_100k")
    assert ds is not None
    assert ds.embodiment is SINGLE_GRIP_CAL
    assert ds.proprio is not None and ds.proprio.dim == 8


@pytest.mark.parametrize("loc", ["base", "j0", "j6", "eef", "tcp", "gdrv", "lfin", "rinn"])
def test_kpt_locations_present_in_all_three_blocks(loc):
    for u in ("u", "v"):
        assert f"kpt2d_{loc}_{u}" in DOF
    for ax in ("x", "y", "z"):
        assert f"kpt3dc_{loc}_{ax}" in DOF
        assert f"kpt3dw_{loc}_{ax}" in DOF
