"""
Serializers to save the user code and wrap it in the Connect algo code.
"""
from connectlib.remote.serializers.pickle_serializer import PickleSerializer
from connectlib.remote.serializers.serializer import Serializer

__all__ = ["Serializer", "PickleSerializer"]
