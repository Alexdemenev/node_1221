"""Top-level package for node_1221."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Alex"""
__email__ = "demenev@1221systems.com"
__version__ = "0.0.1"

from .src.node_1221.nodes import NODE_CLASS_MAPPINGS
from .src.node_1221.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
