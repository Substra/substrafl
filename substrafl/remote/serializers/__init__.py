"""
Serializers to save the user code and wrap it in the Substra algo code.
"""
from substrafl.remote.serializers.pickle_serializer import PickleSerializer
from substrafl.remote.serializers.serializer import Serializer

__all__ = ["Serializer", "PickleSerializer"]
