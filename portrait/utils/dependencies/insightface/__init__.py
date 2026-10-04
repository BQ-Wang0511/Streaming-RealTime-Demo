# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import
from importlib.util import find_spec

if find_spec("onnxruntime") is None:
    raise ImportError(
        "Unable to import dependency onnxruntime. "
    )

__version__ = '0.7.3'

from . import model_zoo as model_zoo
from . import app as app
