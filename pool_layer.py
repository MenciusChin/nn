"""
Pooling layers.
"""

import numpy as np
from layer import Layer
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool