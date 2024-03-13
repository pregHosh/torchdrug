import logging
import sys

from . import patch
from .data.constant import *

logger = logging.getLogger("")
logger.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(format)
logger.addHandler(handler)

__version__ = "1.0.1"
