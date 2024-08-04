# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from joblib import Parallel, delayed
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure


def calculate_dissimilarity(
    anchor_structure: Structure, *structure: Structure, n_jobs=10
):
    """Calculate the dissimilarity between the anchor structure and the other structures.

    Args:
        anchor_structure: The anchor structure.
        structure: The other structures.
        n_jobs: The number of jobs to run in parallel. Default is 10.

    Returns:
        np.array: The dissimilarity between the anchor structure and the other structures.
    """
    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
        stats=("mean", "std_dev", "minimum", "maximum"),
    )
    v_anchor = np.array(ssf.featurize(anchor_structure))
    tmp = Parallel(n_jobs=n_jobs)(delayed(ssf.featurize)(s) for s in structure)
    return [np.linalg.norm(np.array(s) - v_anchor) for s in tmp]


def convert_struct(structure: Structure, volume=None) -> Structure:
    """
    Convert the structure to a primitive structure and adjust the volume of the unit cell.

    Args:
        structure (Structure): The input structure.
        volume (float): The volume of the unit cell. Default is None.

    Returns:
        Structure: The modified structure.
    """
    structure = structure.get_primitive_structure()
    if volume is not None:
        structure.scale_lattice(volume)
    return structure
