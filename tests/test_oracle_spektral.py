"""Orakel-Test spektrale-schnitte-demo: Schnittmaße, Aufzählung, Spektrum, Fiedler-Vektor, Sweep-Cut und Schranken von Cheeger gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12).

Unabhängig: `networkx` (`cut_size`, `normalized_cut_size`, `conductance`, `laplacian_matrix`, `normalized_laplacian_matrix`, `fiedler_vector`, `edge_connectivity`), `numpy.linalg.eigvalsh` auf den Matrizen von
networkx, die Aufzählung mit `itertools.combinations` und die Ungleichung von Cheeger als Eigenschaft. Feste Netze (Seeds)."""

import itertools
import math

import numpy as np
import pytest

import spk_algorithm as A
import spk_constants as C
import spk_scenario as S

nx = pytest.importorskip("networkx")
pytest.importorskip("scipy")

from conftest import MINI, adj_of  # noqa: E402


def _nets(count=36, base=2000):
    kinds = ("random", "grid", "ring", "scalefree", "verbund", "geometric")
    for k in range(count):
        yield S.make_instance(kinds[k % len(kinds)], 8 + (k * 3) % 9, (3.0, 4.5, 6.0)[k % 3], 1 + k % 3, base + k)


def _graph(inst):
    g = nx.Graph()
    g.add_nodes_from(range(inst.n))
    g.add_edges_from(inst.pairs)
    return g


def _brute(inst):
    """Kleinster Wert und Zahl der besten Schnitte je Ziel über alle Teilmengen (jede Zweiteilung zweimal, daher halbiert), mit den Maßen von networkx."""
    g = _graph(inst)
    values = {key: [] for key in C.OBJECTIVES}
    for k in range(1, inst.n):
        for sub in itertools.combinations(range(inst.n), k):
            s = set(sub)
            c = nx.cut_size(g, s)
            values["cut"].append(c)
            values["ratio"].append(c / len(s) + c / (inst.n - len(s)))
            values["ncut"].append(nx.normalized_cut_size(g, s))
            values["cond"].append(nx.conductance(g, s))
    out = {}
    for key, vals in values.items():
        best = min(vals)
        out[key] = (best, sum(1 for v in vals if v <= best + 1e-9) // 2)
    return out


def test_the_oracle_itself_on_the_hand_computed_mini_networks():
    for name, spec in MINI.items():
        inst = S.Instance(spec["n"], tuple(sorted(tuple(sorted(p)) for p in spec["pairs"])), np.zeros((spec["n"], 2)), "textbook")
        got = _brute(inst)
        for key in C.OBJECTIVES:
            assert got[key][0] == pytest.approx(spec[key][0]) and got[key][1] == spec[key][1], (name, key)


def test_exact_optima_equal_the_independent_enumeration():
    for inst in _nets(18):
        if inst.n > 12:
            continue
        want = _brute(inst)
        ex = A.exact_optima(inst.adj)
        for key in C.OBJECTIVES:
            assert ex.opt[key].value == pytest.approx(want[key][0]) and ex.opt[key].count == want[key][1], (inst.kind, inst.n, key)


def test_the_measures_equal_the_networkx_functions_on_random_sets():
    import random
    rng = random.Random(3)
    for inst in _nets(24):
        g = _graph(inst)
        for _ in range(12):
            s = {v for v in range(inst.n) if rng.random() < 0.4}
            if not s or len(s) == inst.n:
                continue
            c, r, nc, co = A.measures(inst.adj, s)
            assert c == nx.cut_size(g, s) and nc == pytest.approx(nx.normalized_cut_size(g, s)) and co == pytest.approx(nx.conductance(g, s))
            assert r == pytest.approx(c / len(s) + c / (inst.n - len(s)))


def test_eigenvalues_equal_numpy_on_the_networkx_matrices():
    for inst in _nets():
        g = _graph(inst)
        for kind, matrix in (("norm", nx.normalized_laplacian_matrix(g)), ("plain", nx.laplacian_matrix(g))):
            want = np.linalg.eigvalsh(matrix.toarray().astype(float))
            sp = A.spectrum(inst.adj, kind)
            assert np.allclose(sp.values, want, atol=1e-9), (inst.kind, kind)
            assert sp.lam2 == pytest.approx(want[1]) and sp.multiplicity == sum(1 for w in want[1:] if abs(w - want[1]) < 1e-8)


def test_the_fiedler_vector_equals_networkx_up_to_sign_when_lambda2_is_simple():
    seen = 0
    for inst in _nets(36, 2200):
        g = _graph(inst)
        for kind, normalized in (("plain", False), ("norm", True)):
            sp = A.spectrum(inst.adj, kind)
            if sp.multiplicity != 1:
                continue
            ref = np.array(nx.fiedler_vector(g, normalized=normalized, method="tracemin_lu", seed=1))
            mine = np.array(sp.vec)
            ref, mine = ref / np.linalg.norm(ref), mine / np.linalg.norm(mine)
            assert min(np.linalg.norm(ref - mine), np.linalg.norm(ref + mine)) < 1e-5, (inst.kind, kind)
            seen += 1
    assert seen >= 30


def test_cheeger_inequality_and_the_sweep_guarantee_on_many_networks():
    for inst in _nets(36, 2400):
        if inst.n > 14:
            continue
        ex = A.exact_optima(inst.adj)
        h = ex.opt["cond"].value
        sp = A.spectrum(inst.adj, "norm")
        lo, hi = A.cheeger_bounds(sp.lam2)
        assert lo <= h + 1e-9 and h <= hi + 1e-9, (inst.kind, inst.n)
        phi = min(A.sweep(inst.adj, sp.x).cond)
        assert h - 1e-9 <= phi <= hi + 1e-9


def test_every_prefix_of_the_sweep_equals_the_networkx_measures():
    for inst in _nets(12, 2600):
        g = _graph(inst)
        sw = A.sweep(inst.adj, A.spectrum(inst.adj, "norm").x)
        for k in range(1, inst.n):
            s = set(sw.order[:k])
            assert sw.cut[k - 1] == nx.cut_size(g, s) and sw.cond[k - 1] == pytest.approx(nx.conductance(g, s)) and sw.ncut[k - 1] == pytest.approx(nx.normalized_cut_size(g, s))


def test_min_cut_equals_the_edge_connectivity():
    for inst in _nets():
        assert A.min_cut(inst.adj)[0] == nx.edge_connectivity(_graph(inst))


def test_known_closed_forms_for_cycles_and_complete_graphs():
    """Geschlossene Formeln: Kreis C_n normalisiert λ₂ = 1 − cos(2π/n) (zweifach), vollständiges K_n normalisiert n/(n−1) (n−1-fach), Pfad P_n unnormalisiert 2 − 2 cos(π/n)."""
    for n in (5, 8, 12):
        cyc = adj_of(n, [(v, (v + 1) % n) for v in range(n)])
        sp = A.spectrum(cyc, "norm")
        assert sp.lam2 == pytest.approx(1 - math.cos(2 * math.pi / n)) and sp.multiplicity == 2
        path = adj_of(n, [(v, v + 1) for v in range(n - 1)])
        assert A.spectrum(path, "plain").lam2 == pytest.approx(2 - 2 * math.cos(math.pi / n))
        complete = adj_of(n, [(a, b) for a in range(n) for b in range(a + 1, n)])
        sp = A.spectrum(complete, "norm")
        assert sp.lam2 == pytest.approx(n / (n - 1)) and sp.multiplicity == n - 1
