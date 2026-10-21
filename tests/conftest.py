"""Gemeinsame Fixtures: die Mini-Netze, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Teure Experimente laufen einmal je Testlauf (`lru_cache`) und werden von Claims-, Verdikt- und Mutationstests gemeinsam genutzt."""

import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def adj_of(n, pairs):
    nb = [set() for _ in range(n)]
    for u, v in pairs:
        nb[u].add(v)
        nb[v].add(u)
    return tuple(tuple(sorted(s)) for s in nb)


# Von Hand gerechnete Mini-Netze. Je Ziel: (kleinster Wert, Zahl der Schnitte mit diesem Wert); `total` = Zahl aller Schnitte = 2^(n-1) - 1; `vol` = 2m;
# `lam2` und `mult` für die normalisierte Matrix.
MINI = {
    "zwei_dreiecke": dict(n=6, pairs=[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)], total=31, vol=14,
                          cut=(1, 1), ratio=(2 / 3, 1), ncut=(2 / 7, 1), cond=(1 / 7, 1)),
    "pfad4": dict(n=4, pairs=[(0, 1), (1, 2), (2, 3)], total=7, vol=6,
                  cut=(1, 3), ratio=(1.0, 1), ncut=(2 / 3, 1), cond=(1 / 3, 1), lam2=0.5, mult=1),
    "kreis6": dict(n=6, pairs=[(v, (v + 1) % 6) for v in range(6)], total=31, vol=12,
                   cut=(2, 15), ratio=(4 / 3, 3), ncut=(2 / 3, 3), cond=(1 / 3, 3), lam2=0.5, mult=2),
    "vollstaendig4": dict(n=4, pairs=[(a, b) for a in range(4) for b in range(a + 1, 4)], total=7, vol=12,
                          cut=(3, 4), ratio=(4.0, 7), ncut=(4 / 3, 7), cond=(2 / 3, 3), lam2=4 / 3, mult=3),
    "stern5": dict(n=5, pairs=[(0, v) for v in range(1, 5)], total=15, vol=8,
                   cut=(1, 4), ratio=(1.25, 4), ncut=(8 / 7, 4), cond=(1.0, 15), lam2=1.0, mult=3),
    "hantel": dict(n=8, pairs=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 4), (0, 7)], total=127, vol=22,
                   cut=(1, 2), ratio=(8 / 15, 1), ncut=(22 / 105, 1), cond=(1 / 7, 1)),
}


class FixedRandom:
    """Liefert immer fast 1: Fisher–Yates lässt die Reihenfolge unverändert."""

    def random(self):
        return 0.999999


@pytest.fixture
def mini():
    return MINI


@lru_cache(maxsize=None)
def cached(name):
    """Einmal je Testlauf berechnete Experimente (Name der Funktion in spk_experiments)."""
    import spk_experiments as ex
    return getattr(ex, name)()
