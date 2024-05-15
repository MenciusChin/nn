"""
Helper functions, for now
"""

import numpy as np
from layer import Layer
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool


def itot(param):
    """
    This reformat the input parameter,
    returning (int, int)
    """

    return (param, param) if isinstance(param, int) else param