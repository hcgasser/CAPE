""" This module specifies the immune-visibility profiles. """

from enum import Enum


class Profile(Enum):
    """The immune-visibility profiles."""

    BASE = "baseline"
    VIS_DOWN = "reduced"
    VIS_UP = "increased"
    VIS_UP_NAT = "inc-nat"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def reward_visible_natural(self):
        """Returns the reward for naturally visible epitopes"""

        if self == Profile.VIS_DOWN:
            return -1.0
        if self == Profile.VIS_UP:
            return 1.0
        if self == Profile.VIS_UP_NAT:
            return 1.0
        if self == Profile.BASE:
            return 0.0

        raise ValueError(f"unknown profile {self}")

    def reward_visible_artificial(self):
        """Returns the reward for artificially visible epitopes"""

        if self == Profile.VIS_DOWN:
            return -1.0
        if self == Profile.VIS_UP:
            return 1.0
        if self == Profile.VIS_UP_NAT:
            return -1.0
        if self == Profile.BASE:
            return 0.0
        raise ValueError(f"unknown profile {self}")
