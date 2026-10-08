"""Event-keyed random streams.

Every random draw is keyed by (world seed, stream name, entity, day, ...), not by
the order in which code consumes randomness. Two branches that take different
actions therefore still see the same exogenous draws (market, customer orders),
and the same purchase made in two branches gets the same lot draws.
"""

from __future__ import annotations

import hashlib

import numpy as np


def key_seed(*parts) -> int:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def stream(*parts) -> np.random.Generator:
    return np.random.default_rng(key_seed(*parts))
