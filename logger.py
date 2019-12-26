import logging
import sys
from logging import INFO, Formatter, StreamHandler

"""
Setting basic logging configuration for other functions to use.

We log at the Info level for auditing model training/evaluation,
and stream directly to stdout so the user can pipe to a destination
of their choosing.
"""

logger = logging.getLogger(__name__)
logger.setLevel(INFO)

# formatting 

log_format = '%(asctime)s [%(levelname)s] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
formatter = Formatter(fmt=log_format, datefmt=date_format)

# log handlers

handler = StreamHandler(sys.stdout)
handler.setFormatter(formatter)

# modifying logger object

logger.addHandler(handler)

