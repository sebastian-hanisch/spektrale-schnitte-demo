"""Kern: Schnittmaße, Aufzählung aller Schnitte, Spektrum der Laplace-Matrix, Fiedler-Vektor, Sweep-Cut und die Schranken von Cheeger.

Ein **Schnitt** teilt die Standorte in eine Menge S und ihren Rest; sein Wert c(S) ist die Zahl der Leitungen dazwischen. Mit n Standorten, m Leitungen, Grad d(v), Volumen vol(S) = Σ d(v) über S (und vol(V) = 2m):

* RatioCut(S) = c / |S| + c / (n − |S|)
* Ncut(S)     = c / vol(S) + c / (vol(V) − vol(S))
* Konduktanz φ(S) = c / min(vol(S), vol(V) − vol(S)); die kleinste heißt Cheeger-Konstante h.

Die normalisierte Laplace-Matrix L = I − D^(−1/2) A D^(−1/2) hat den Eigenwert 0 (Vektor D^(1/2)·1) und als zweitkleinsten λ₂; für zusammenhängende Netze ist λ₂ > 0 und
**λ₂ / 2 ≤ h ≤ √(2 λ₂)** (Cheeger). Der **Sweep-Cut** sortiert die Standorte nach dem Fiedler-Vektor (D^(−1/2) · Eigenvektor zu λ₂) und nimmt den besten Anfang der Reihenfolge; sein φ liegt zwischen h und √(2 λ₂).

Ist λ₂ mehrfach (z. B. beim Gitter), ist der Fiedler-Vektor nicht eindeutig; die Demo nimmt dann die Projektion eines festen Hilfsvektors auf den Eigenraum (`C.SIGNATURE`), unabhängig von der Basis, die die Bibliothek wählt."""

import math
from dataclasses import dataclass

import numpy as np

import spk_constants as C


# --- Schnittmaße ---------------------------------------------------------------------------------------------------------------------------------


def degrees(adj):
    return tuple(len(a) for a in adj)


def cut_size(adj, side):
    """Zahl der Leitungen zwischen `side` und dem Rest."""
    s = set(side)
    return sum(1 for u in s for v in adj[u] if v not in s)


def volume(adj, side):
    return sum(len(adj[v]) for v in side)


def ratio_cut(cut, size, n):
    return cut / size + cut / (n - size)


def ncut(cut, vol_side, vol_total):
    return cut / vol_side + cut / (vol_total - vol_side)


def conductance(cut, vol_side, vol_total):
    return cut / min(vol_side, vol_total - vol_side)


def measures(adj, side):
    """(Wert, RatioCut, Ncut, Konduktanz) einer Menge."""
    n, total = len(adj), sum(len(a) for a in adj)
    s = set(side)
    c, vol = cut_size(adj, s), volume(adj, s)
    return c, ratio_cut(c, len(s), n), ncut(c, vol, total), conductance(c, vol, total)


def is_trivial(n, side_set):
    """Trennt der Schnitt nur einen einzelnen Standort ab?"""
    return len(side_set) == 1 or len(side_set) == n - 1


def cheeger_bounds(lam2):
    return lam2 / 2, math.sqrt(2 * lam2)


# --- Aufzählung aller Schnitte (n <= BRUTE_N) ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Optimum:
    value: float
    count: int              # Zahl der Schnitte mit diesem Wert (jede Zweiteilung einmal)
    masks: tuple            # ihre Mengen S (ohne den Standort n − 1), als Bitmasken
    side: tuple             # eine davon (die erste), als Standorte, die kleinere Seite zuerst
    all_trivial: bool       # trennt jeder dieser Schnitte nur einen Standort ab?
    any_trivial: bool


@dataclass(frozen=True)
class Exact:
    total: int                  # Zahl aller Schnitte, 2^(n−1) − 1
    opt: dict                   # Ziel -> Optimum, Ziele `C.OBJECTIVES`

    def same(self, a, b):
        """Haben die Ziele a und b genau dieselben besten Schnitte?"""
        return self.opt[a].masks == self.opt[b].masks


def _bit(masks, v, n):
    return (masks >> v) & 1 if v < n - 1 else np.zeros_like(masks)


def enumerate_cuts(adj):
    """Alle 2^(n−1) − 1 Zweiteilungen {S, Rest} mit dem Standort n − 1 im Rest: Arrays (Masken, Schnitt, Größe, Volumen)."""
    n = len(adj)
    masks = np.arange(1, 1 << (n - 1), dtype=np.int64)
    cut = np.zeros(len(masks), dtype=np.int64)
    size = np.zeros(len(masks), dtype=np.int64)
    vol = np.zeros(len(masks), dtype=np.int64)
    for u in range(n):
        bu = _bit(masks, u, n)
        size += bu
        vol += bu * len(adj[u])
        for v in adj[u]:
            if u < v:
                cut += bu ^ _bit(masks, v, n)
    return masks, cut, size, vol


def _side_of(mask, n):
    s = tuple(v for v in range(n - 1) if (int(mask) >> v) & 1)
    rest = tuple(v for v in range(n) if v not in s)
    return s if len(s) < len(rest) or (len(s) == len(rest) and 0 in s) else rest


def exact_optima(adj):
    """Kleinster Wert, Zahl und Gestalt der besten Schnitte je Ziel (Schnitt ohne Balance, RatioCut, Ncut, Konduktanz) durch Aufzählung."""
    n = len(adj)
    masks, cut, size, vol = enumerate_cuts(adj)
    total = sum(len(a) for a in adj)
    c = cut.astype(float)
    values = {
        "cut": c,
        "ratio": c / size + c / (n - size),
        "ncut": c / vol + c / (total - vol),
        "cond": c / np.minimum(vol, total - vol),
    }
    trivial = (size == 1) | (size == n - 1)
    opt = {}
    for key, arr in values.items():
        best = float(arr.min())
        sel = np.flatnonzero(arr <= best + C.TOL * max(1.0, abs(best)))
        opt[key] = Optimum(best, len(sel), tuple(int(masks[i]) for i in sel), _side_of(masks[sel[0]], n), bool(trivial[sel].all()), bool(trivial[sel].any()))
    return Exact(len(masks), opt)


# --- Spektrum ------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    kind: str                   # "norm" oder "plain"
    values: tuple               # alle Eigenwerte aufsteigend
    lam2: float
    multiplicity: int           # Vielfachheit von λ₂ (1: der Fiedler-Vektor ist eindeutig)
    x: tuple                    # Fiedler-Vektor: D^(−1/2)·v (normalisiert) bzw. v (unnormalisiert), v der Eigenvektor zu λ₂
    vec: tuple                  # der Eigenvektor v selbst, Länge 1


def adjacency(adj):
    n = len(adj)
    a = np.zeros((n, n))
    for u in range(n):
        for v in adj[u]:
            a[u, v] = 1.0
    return a


def laplacian(adj, kind):
    a = adjacency(adj)
    d = a.sum(axis=1)
    if kind == "norm":
        s = 1.0 / np.sqrt(d)
        return np.eye(len(adj)) - a * s[:, None] * s[None, :]
    if kind == "plain":
        return np.diag(d) - a
    raise ValueError(f"unbekannte Matrix {kind}")


def helper_vector(n):
    """Fester Hilfsvektor (plattformstabil), der in einem entarteten Eigenraum den Fiedler-Vektor festlegt."""
    return np.array([math.sin(C.SIGNATURE[0] * (i + 1)) + math.cos(C.SIGNATURE[1] * (i + 1) ** 2) for i in range(n)])


def spectrum(adj, kind):
    n = len(adj)
    vals, vecs = np.linalg.eigh(laplacian(adj, kind))
    lam2 = float(vals[1])
    idx = [i for i in range(1, n) if vals[i] - lam2 <= C.TOL * max(1.0, abs(lam2))]
    sub = vecs[:, idx]
    v = sub @ (sub.T @ helper_vector(n))
    norm = float(np.linalg.norm(v))
    v = sub[:, 0] if norm < 1e-12 else v / norm
    x = v / np.sqrt(np.array([len(a) for a in adj], dtype=float)) if kind == "norm" else v
    return Spectrum(kind, tuple(float(t) for t in vals), lam2, len(idx), tuple(float(t) for t in x), tuple(float(t) for t in v))


# --- Sweep-Cut -----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    order: tuple                # Standorte nach dem Fiedler-Vektor aufsteigend (Gleichstand: kleinere Nummer)
    cut: tuple                  # für jeden Anfang der Länge k = 1 .. n − 1
    size: tuple
    vol: tuple
    ratio: tuple
    ncut: tuple
    cond: tuple

    def series(self, measure):
        return {"cut": self.cut, "ratio": self.ratio, "ncut": self.ncut, "cond": self.cond}[measure]

    def best(self, measure):
        """Länge des besten Anfangs (der erste mit dem kleinsten Wert) für das Maß."""
        s = self.series(measure)
        low = min(s)
        return next(k + 1 for k, t in enumerate(s) if t <= low + C.TOL * max(1.0, abs(low)))

    def prefix(self, k):
        return tuple(sorted(self.order[:k]))


def sweep(adj, x):
    n = len(adj)
    order = tuple(sorted(range(n), key=lambda i: (round(x[i], 9), i)))
    total = sum(len(a) for a in adj)
    inside = set()
    cut = vol = 0
    cuts, sizes, vols, ratios, ncuts, conds = [], [], [], [], [], []
    for k, v in enumerate(order[:-1], start=1):
        cut += len(adj[v]) - 2 * sum(1 for u in adj[v] if u in inside)
        inside.add(v)
        vol += len(adj[v])
        cuts.append(cut)
        sizes.append(k)
        vols.append(vol)
        ratios.append(ratio_cut(cut, k, n))
        ncuts.append(ncut(cut, vol, total))
        conds.append(conductance(cut, vol, total))
    return Sweep(order, tuple(cuts), tuple(sizes), tuple(vols), tuple(ratios), tuple(ncuts), tuple(conds))


# --- Kleinster Schnitt ohne Balance für große Netze ----------------------------------------------------------------------------------------------


def min_cut(adj):
    """Kleinster Schnitt ohne Balance (Stoer–Wagner, ungewichtet): (Wert, eine Seite)."""
    n = len(adj)
    if n < 2:
        return 0, ()
    w = [[1 if v in adj[u] else 0 for v in range(n)] for u in range(n)]
    members = {v: [v] for v in range(n)}
    active = list(range(n))
    best = None
    while len(active) > 1:
        key = {v: 0 for v in active}
        rest = list(active)
        order, bind = [], []
        while rest:
            pick = min(rest, key=lambda v: (-key[v], v))
            order.append(pick)
            bind.append(key[pick])
            rest.remove(pick)
            for v in rest:
                key[v] += w[pick][v]
        last, before = order[-1], order[-2]
        if best is None or bind[-1] < best[0]:
            best = (bind[-1], tuple(sorted(members[last])))
        for x in active:
            if x != last and x != before:
                w[before][x] += w[last][x]
                w[x][before] = w[before][x]
        members[before].extend(members[last])
        active.remove(last)
    return best
