"""Kern: jede Einheit an von Hand gerechneten Mini-Netzen (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import spk_algorithm as A
import spk_constants as C

from conftest import MINI, adj_of


def _adj(name):
    return adj_of(MINI[name]["n"], MINI[name]["pairs"])


def test_the_three_measures_by_hand():
    assert A.ratio_cut(2, 3, 8) == pytest.approx(2 / 3 + 2 / 5)
    assert A.ncut(2, 4, 12) == pytest.approx(2 / 4 + 2 / 8)
    assert A.conductance(2, 4, 12) == pytest.approx(2 / 4) and A.conductance(2, 9, 12) == pytest.approx(2 / 3)          # das kleinere Volumen zählt
    assert A.cheeger_bounds(0.5) == (0.25, 1.0) and A.cheeger_bounds(2.0)[1] == 2.0


def test_cut_volume_and_measures_of_a_set_by_hand():
    adj = _adj("zwei_dreiecke")
    assert A.degrees(adj) == (2, 2, 3, 3, 2, 2)
    assert A.cut_size(adj, {0, 1, 2}) == 1 and A.cut_size(adj, (3, 4, 5)) == 1 and A.cut_size(adj, {0, 3}) == 5 and A.cut_size(adj, set()) == 0
    assert A.volume(adj, {0, 1, 2}) == 7 and A.volume(adj, {2}) == 3
    c, r, nc, co = A.measures(adj, {0, 1, 2})
    assert (c, r, nc, co) == (1, pytest.approx(2 / 3), pytest.approx(2 / 7), pytest.approx(1 / 7))


def test_is_trivial_boundaries():
    assert A.is_trivial(5, {2}) and A.is_trivial(5, {0, 1, 2, 3}) and not A.is_trivial(5, {0, 1}) and not A.is_trivial(5, {0, 1, 2}) and not A.is_trivial(4, {0, 1})


@pytest.mark.parametrize("name", list(MINI))
def test_exact_optima_of_every_mini_network(name):
    spec = MINI[name]
    ex = A.exact_optima(_adj(name))
    assert ex.total == spec["total"]
    for key in C.OBJECTIVES:
        value, count = spec[key]
        o = ex.opt[key]
        assert o.value == pytest.approx(value) and o.count == count and len(o.masks) == count, (name, key)
        assert 0 < len(o.side) < spec["n"] and len(o.side) <= spec["n"] - len(o.side)               # die kleinere Seite zuerst


def test_exact_optima_sides_and_trivial_flags_by_hand():
    ex = A.exact_optima(_adj("hantel"))
    assert ex.opt["ratio"].side == (4, 5, 6) and ex.opt["ncut"].side == (4, 5, 6) and ex.opt["cond"].side == (4, 5, 6)         # {E, F, G} gegen den Rest
    assert not ex.opt["cut"].all_trivial and ex.opt["cut"].any_trivial and not ex.opt["ncut"].any_trivial                 # kleinster Schnitt: {H} oder {E, F, G}
    pfad = A.exact_optima(_adj("pfad4"))
    assert pfad.opt["cut"].any_trivial and not pfad.opt["cut"].all_trivial and pfad.opt["cond"].side == (0, 1) and not pfad.opt["cond"].any_trivial
    assert A.exact_optima(_adj("stern5")).opt["cut"].all_trivial and A.exact_optima(_adj("vollstaendig4")).opt["cut"].all_trivial
    assert not A.exact_optima(_adj("vollstaendig4")).opt["cond"].any_trivial
    assert A.exact_optima(_adj("zwei_dreiecke")).opt["cut"].side == (0, 1, 2)                                              # Gleichstand der Größe: die Seite mit dem Standort 0


def test_same_optimal_cuts_by_hand():
    assert A.exact_optima(_adj("kreis6")).same("ncut", "ratio") and A.exact_optima(_adj("kreis6")).same("ncut", "cond")
    k4 = A.exact_optima(_adj("vollstaendig4"))
    assert k4.same("ratio", "ncut") and not k4.same("ncut", "cond")
    assert A.exact_optima(_adj("stern5")).same("ratio", "ncut") and not A.exact_optima(_adj("stern5")).same("ncut", "cond")


def test_enumerate_cuts_by_hand_on_the_path():
    masks, cut, size, vol = A.enumerate_cuts(_adj("pfad4"))
    assert list(masks) == [1, 2, 3, 4, 5, 6, 7]
    assert list(cut) == [1, 2, 1, 2, 3, 2, 1]                                                  # {0} {1} {0,1} {2} {0,2} {1,2} {0,1,2}
    assert list(size) == [1, 1, 2, 1, 2, 2, 3] and list(vol) == [1, 2, 3, 2, 3, 4, 5]


def test_laplacian_matrices_by_hand():
    adj = _adj("pfad4")
    plain = A.laplacian(adj, "plain")
    assert plain.tolist() == [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]]
    norm = A.laplacian(adj, "norm")
    assert norm[0, 0] == 1 and norm[0, 1] == pytest.approx(-1 / math.sqrt(2)) and norm[1, 2] == pytest.approx(-0.5) and norm[0, 2] == 0
    with pytest.raises(ValueError):
        A.laplacian(adj, "signless")
    assert A.adjacency(adj).tolist() == [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


@pytest.mark.parametrize("name", [k for k in MINI if "lam2" in MINI[k]])
def test_normalized_lambda2_and_multiplicity_by_hand(name):
    sp = A.spectrum(_adj(name), "norm")
    assert sp.lam2 == pytest.approx(MINI[name]["lam2"]) and sp.multiplicity == MINI[name]["mult"]
    assert sp.values[0] == pytest.approx(0.0, abs=1e-12) and list(sp.values) == sorted(sp.values)


def test_plain_lambda2_by_hand():
    assert A.spectrum(_adj("pfad4"), "plain").lam2 == pytest.approx(2 - math.sqrt(2))               # 2 − 2 cos(π/4)
    sp = A.spectrum(_adj("kreis6"), "plain")
    assert sp.lam2 == pytest.approx(1.0) and sp.multiplicity == 2                                    # 2 − 2 cos(π/3)
    assert A.spectrum(_adj("vollstaendig4"), "plain").lam2 == pytest.approx(4.0)


def test_the_fiedler_vector_is_an_eigenvector_normalized_and_orthogonal_to_the_trivial_one():
    for name in MINI:
        adj = _adj(name)
        for kind in C.LAPS:
            sp = A.spectrum(adj, kind)
            v = np.array(sp.vec)
            assert float(np.linalg.norm(v)) == pytest.approx(1.0)
            assert np.allclose(A.laplacian(adj, kind) @ v, sp.lam2 * v, atol=1e-9)
            trivial = np.sqrt(np.array(A.degrees(adj), dtype=float)) if kind == "norm" else np.ones(len(adj))
            assert abs(float(np.dot(trivial, v))) < 1e-9
            expect = v / np.sqrt(np.array(A.degrees(adj), dtype=float)) if kind == "norm" else v
            assert np.allclose(sp.x, expect)


def test_the_fiedler_vector_of_the_path_is_antisymmetric():
    sp = A.spectrum(_adj("pfad4"), "plain")
    x = sp.x
    assert x[0] == pytest.approx(-x[3]) and x[1] == pytest.approx(-x[2]) and abs(x[0]) > abs(x[1])
    assert x[0] * x[1] > 0 and x[1] * x[2] < 0
    assert A.spectrum(_adj("pfad4"), "plain") == sp                                                   # deterministisch


def test_the_helper_vector_fixes_the_vector_in_a_degenerate_eigenspace():
    sp = A.spectrum(_adj("vollstaendig4"), "norm")
    assert sp.multiplicity == 3
    h = A.helper_vector(4)
    v = np.array(sp.vec)
    sub = np.linalg.eigh(A.laplacian(_adj("vollstaendig4"), "norm"))[1][:, 1:]
    assert np.allclose(v, sub @ (sub.T @ h) / np.linalg.norm(sub @ (sub.T @ h)))                    # Projektion des Hilfsvektors, unabhängig von der Basis
    assert not np.allclose(A.helper_vector(5)[:4], A.helper_vector(4)[:2].tolist() * 2)               # der Hilfsvektor ist nicht periodisch


def test_the_helper_vector_by_hand():
    """sin(π/10 · k) für k = 1 .. 4 ist sin 18°, 36°, 54°, 72°; dazu cos(1,2345678901 · k²)."""
    h = A.helper_vector(4)
    assert list(h) == pytest.approx([0.309017 + 0.329929, 0.587785 + 0.223967, 0.809017 + 0.115280, 0.951057 + 0.618841], abs=1e-6)
    assert list(A.helper_vector(2)) == pytest.approx(list(h[:2]))                                  # die ersten Einträge hängen nicht von n ab


def test_sweep_of_a_given_vector_by_hand():
    adj = _adj("zwei_dreiecke")
    sw = A.sweep(adj, (-3, -2, -1, 1, 2, 3))
    assert sw.order == (0, 1, 2, 3, 4, 5)
    assert sw.cut == (2, 2, 1, 2, 2) and sw.size == (1, 2, 3, 4, 5) and sw.vol == (2, 4, 7, 10, 12)
    assert sw.cond[2] == pytest.approx(1 / 7) and sw.ncut[2] == pytest.approx(2 / 7) and sw.ratio[2] == pytest.approx(2 / 3) and sw.cond[0] == pytest.approx(1.0)
    assert sw.cond[3] == pytest.approx(2 / 4) and sw.ncut[3] == pytest.approx(2 / 10 + 2 / 4)                  # k = 4: Volumen 10 gegen 4
    assert sw.best("cut") == sw.best("cond") == sw.best("ncut") == sw.best("ratio") == 3
    assert sw.prefix(3) == (0, 1, 2) and sw.series("cut") is sw.cut


def test_sweep_order_follows_the_vector_and_breaks_ties_by_index():
    adj = _adj("zwei_dreiecke")
    assert A.sweep(adj, (3, 2, 1, -1, -2, -3)).order == (5, 4, 3, 2, 1, 0)
    assert A.sweep(adj, (0, 0, 0, 0, 0, 0)).order == (0, 1, 2, 3, 4, 5)
    assert A.sweep(adj, (1e-12, 0, 0, 0, 0, 0)).order == (0, 1, 2, 3, 4, 5)                           # Unterschiede unter 1e-9 gelten als Gleichstand
    assert A.sweep(adj, (1e-6, 0, 0, 0, 0, 0)).order == (1, 2, 3, 4, 5, 0)


def test_sweep_best_is_the_first_minimum():
    adj = _adj("kreis6")
    sw = A.sweep(adj, (0, 1, 2, 3, 4, 5))                                                              # Anfang der Länge 3: Bogen aus drei Standorten, Schnitt 2
    assert sw.cut == (2, 2, 2, 2, 2) and sw.best("cut") == 1 and sw.best("cond") == 3 and sw.best("ratio") == 3


def test_sweep_of_the_path_is_independent_of_the_sign_of_the_vector():
    sw = A.sweep(_adj("pfad4"), A.spectrum(_adj("pfad4"), "norm").x)
    assert sw.cut == (1, 1, 1) and sw.vol == (1, 3, 5)
    assert sw.ratio == pytest.approx((4 / 3, 1.0, 4 / 3)) and sw.ncut == pytest.approx((1.2, 2 / 3, 1.2)) and sw.cond == pytest.approx((1.0, 1 / 3, 1.0))
    assert sw.best("cond") == 2 and sw.prefix(2) in ((0, 1), (2, 3))


def test_min_cut_of_every_mini_network_and_its_side():
    for name, spec in MINI.items():
        adj = _adj(name)
        value, side = A.min_cut(adj)
        assert value == spec["cut"][0] and A.cut_size(adj, side) == value and 0 < len(side) < spec["n"], name
    assert A.min_cut(adj_of(1, [])) == (0, ())
    assert A.min_cut(_adj("zwei_dreiecke"))[1] in ((0, 1, 2), (3, 4, 5))
    assert A.min_cut(_adj("hantel")) == (1, (7,))                                                      # Stoer–Wagner endet am Standort H
