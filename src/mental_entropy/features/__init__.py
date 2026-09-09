"""Feature extraction modules for Mental Entropy Score (MES).

This package contains all feature extraction modules:
- CE (Coherence Entropy)
- SE (Semantic Entropy)
- NE (Narrative Entropy)
- CLE (Cognitive Load Entropy)
- BC (Belief Conflict)
- GD (Global Disorder)
"""

from mental_entropy.utils.thresholds import (
    BREAK_T,
    CLUSTER_T,
    CONFLICT_SIM_T,
    HIGH_T,
    LOW_T,
    REP_T,
    SHARP_DROP_T,
)

from mental_entropy.features.ce import (
    CE_FEATURE_KEYS,
    ce_features,
    ce_features_from_result,
)
from mental_entropy.features.cle import (
    CLE_FEATURE_KEYS,
    cle_features,
    cle_features_from_result,
)
from mental_entropy.features.ne import (
    NE_FEATURE_KEYS,
    ne_features,
    ne_features_from_result,
)
from mental_entropy.features.se import (
    SE_FEATURE_KEYS,
    se_features,
    se_features_from_result,
)
from mental_entropy.features.bc import (
    BC_FEATURE_KEYS,
    bc_features,
    bc_features_from_result,
)
from mental_entropy.features.gd import (
    GD_FEATURE_KEYS,
    gd_features,
    gd_features_from_result,
)

__all__ = [
    # CE
    "CE_FEATURE_KEYS",
    "BREAK_T",
    "SHARP_DROP_T",
    "ce_features",
    "ce_features_from_result",
    # SE
    "SE_FEATURE_KEYS",
    "CLUSTER_T",
    "se_features",
    "se_features_from_result",
    # NE
    "NE_FEATURE_KEYS",
    "ne_features",
    "ne_features_from_result",
    # CLE
    "CLE_FEATURE_KEYS",
    "LOW_T",
    "HIGH_T",
    "REP_T",
    "cle_features",
    "cle_features_from_result",
    # BC
    "BC_FEATURE_KEYS",
    "CONFLICT_SIM_T",
    "bc_features",
    "bc_features_from_result",
    # GD
    "GD_FEATURE_KEYS",
    "gd_features",
    "gd_features_from_result",
]
