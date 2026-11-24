"""Early-pregnancy fetal cardiac axis pipeline.

Data reading spec: CARDIAC_AXIS_DATA_READING_SPEC_v1.0
Method stage: fixed_geometry (no historical measurement algorithms reused)
"""

SPEC_VERSION = "CARDIAC_AXIS_DATA_READING_SPEC_v1.0"
METHOD_HISTORY_VERSION = "CARDIAC_AXIS_METHOD_HISTORY_v1.0"
CONTRACT_VERSION = "1.0"

# Formal, contract-frozen label semantics (do not redefine).
LABEL_IDS = (1, 4, 5, 6)
CARD_LABEL_IDS = (4, 5)
REF_SPINE_ID = 6   # id6 centroid
REF_CENTER_ID = 1  # id1 centroid
NUM_CLASSES_INCL_BG = 7
