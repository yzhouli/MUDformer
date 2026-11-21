"""Five-block audit inputs with explicit shared real-history eligibility."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from mudformer.data import SyntheticRiskBlockDataset as BaseSyntheticDataset
from mudformer.data import UserFeatureDataset, load_manifest
from mudformer.temporal_localization import (
    inject_tweet_feature_blocks as original_tweet_injection,
    synthetic_training_sequence_injection_records,
)

MIN_BEHAVIORS = 50


class SyntheticRiskBlockDataset(BaseSyntheticDataset):
    """Synthesize only histories with five real behaviors; preserve the model."""
    def __init__(self, target_manifest, donor_manifest, feature_root=None,
                 max_length=4096, auxiliary_features=None):
        UserFeatureDataset.__init__(self, target_manifest, feature_root, max_length,
            training=False, random_crop=False, auxiliary_features=auxiliary_features)
        self.records = [r for r in self.records if r.label == 0 and r.behavior_count >= MIN_BEHAVIORS]
        donors = [r for r in load_manifest(donor_manifest) if r.label == 1 and r.behavior_count >= MIN_BEHAVIORS]
        if not self.records or not donors:
            raise ValueError("five-block injection requires eligible targets and donors")
        if {r.user_id for r in self.records} & {r.user_id for r in donors}:
            raise ValueError("synthetic targets and donors overlap")
        self.donors = {r.user_id: r for r in donors}
        self.plan = {r.user_id: r for r in synthetic_training_sequence_injection_records(
            [r.user_id for r in self.records], [r.user_id for r in donors])}
