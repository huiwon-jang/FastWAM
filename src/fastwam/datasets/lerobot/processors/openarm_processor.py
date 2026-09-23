"""FastWAMProcessor that carries the OpenArm per-sample loss masks through `preprocess`.

`FastWAMProcessor.preprocess` rebuilds the sample dict from a fixed key list, which would drop the flags the
OpenArm dataset attaches (`has_action`, `action_dim_mask`, `is_human`). Everything else is inherited unchanged.
"""
from typing import Any, Dict

from .fastwam_processor import FastWAMProcessor

OPENARM_EXTRA_KEYS = ("has_action", "action_dim_mask", "is_human", "dataset_index")


class OpenArmProcessor(FastWAMProcessor):
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        extras = {k: data[k] for k in OPENARM_EXTRA_KEYS if k in data}
        sample = super().preprocess(data)
        sample.update(extras)
        return sample
