# pylint: disable=wildcard-import
__version__ = "0.1"

from .step_methods import *

import logging
_log = logging.getLogger('pymc3-experimental')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
