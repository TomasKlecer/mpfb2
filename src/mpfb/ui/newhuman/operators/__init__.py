"""Operators for new human."""

from ....services import LogService
_LOG = LogService.get_logger("newhuman.operators")
_LOG.trace("initializing new human operators module")

from .createhuman import MPFB_OT_CreateHumanOperator
from .humanfrompresets import MPFB_OT_HumanFromPresetsOperator
from .humanfrommhm import MPFB_OT_HumanFromMHMOperator

__all__ = [
    "MPFB_OT_CreateHumanOperator",
    "MPFB_OT_HumanFromPresetsOperator",
    "MPFB_OT_HumanFromMHMOperator"
]
