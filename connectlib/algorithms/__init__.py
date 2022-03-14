from connectlib.algorithms.algo import Algo

# This is needed for auto doc to find that Algo module's is algorithms.algo, otherwise when
# trying to link Algo references from one page to the Algo documentation page, it fails.
Algo.__module__ = "algorithms.algo"

__all__ = ["Algo"]
