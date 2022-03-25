"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

from .test import single_gpu_test
from .inference import inference_detector, init_detector, show_result

__all__ = ['single_gpu_test', 'inference_detector', 'show_result', 'init_detector']
