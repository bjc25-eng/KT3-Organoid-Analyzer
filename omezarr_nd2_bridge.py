from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import large_data_core as ldc
from nd2_qc import process_large_experiment_qc

_ORIGINAL_METADATA = None
_ORIGINAL_READ_CHANNEL = None
_ORIGINAL_SCAN = None
SCALING_SCHEMA_VERSION = 'nd2-omezarr-window-scaling-v2'


def _unit_to_um(value: