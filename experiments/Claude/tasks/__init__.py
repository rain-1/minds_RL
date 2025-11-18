"""
Task abstraction system for pluggable cognitive experiments.
"""

from .base import Task
from .character_recall import CharacterRecallTask
from .visualization_recall import VisualizationRecallTask
from .visualization_recall_configurable import VisualizationRecallTaskConfigurable

__all__ = [
    'Task',
    'CharacterRecallTask',
    'VisualizationRecallTask',
    'VisualizationRecallTaskConfigurable'
]
