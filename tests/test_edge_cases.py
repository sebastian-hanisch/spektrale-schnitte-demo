"""Randfälle der Kernroutinen: kleinste Netze, Gleichstände, Entartung, Grenzen inklusive (DEMO-PLAYBOOK §12)."""

import dataclasses

import numpy as np
import pytest

import spk_algorithm as A
import spk_constants as C
import spk_evaluation as ev
import spk_scenario as S

from conftest import adj_of


def test_two_locations_form_the_smallest_network():
    adj = adj_of(2, [(0, 1)])
    ex = A.exact_optima(adj)
    assert ex.total == 1 and ex.opt["cut"].value == 1 and ex.opt["cond"].value == 1.0 and ex.opt["ncut"].value == pytest.approx(2.0) and ex.opt["ratio"].value == pytest.approx(2.0)
    assert ex.opt["cut"].all_trivial and A.min_cut(adj) == (1, (1,))
    sp = A.spectrum(adj, "norm")
    assert sp.lam2 == pytest.approx(2.0) and sp.multiplicity == 1
    sw = A.sweep(adj, sp.x)
    assert sw.cut == (1,) and sw.best("cond") == 1
    lo, hi = A.cheeger_bounds(sp.lam2)
    assert lo <= ex.opt["cond"].value <= hi                                                  # h = 1 liegt genau auf der unteren Schranke


def test_three_locations_on_a_path_and_a_triangle():
    path = adj_of(3, [(0, 1), (1, 2)])
    ex = A.exact_optima(path)
    assert ex.total == 3 and ex.opt["cut"].value == 1 and ex.opt["cut"].count == 2 and ex.opt["cut"].all_trivial
    tri = adj_of(3, [(0, 1), (1, 2), (0, 2)])
    sp = A.spectrum(tri, "norm")
    assert sp.lam2 == pytest.approx(1.5) and sp.multiplicity == 2                          # vollständiges K_3: n / (n − 1), zweifach
    assert A.exact_optima(tri).opt["cut"].value == 2


def test_a_tie_in_the_sweep_values_picks_the_first_and_the_enumeration_counts_all():
    ring = adj_of(8, [(v, (v + 1) % 8) for v in range(8)])
    ex = A.exact_optima(ring)
    assert ex.opt["cond"].count == 4 and ex.opt["cond"].side == (0, 1, 2, 3)                # vier Halbierungen des Rings, die erste enthält den Standort 0
    sw = A.sweep(ring, tuple(range(8)))
    assert sw.best("cond") == 4 and sw.best("cut") == 1                                      # kleinster Schnitt: der erste Anfang mit Wert 2


def test_exactly_two_equal_smaller_sides_are_normalized_to_the_side_with_location_zero():
    assert A._side_of(0b0011, 4) == (0, 1)                                                   # {0, 1} gegen {2, 3}: Gleichstand der Größe
    assert A._side_of(0b0110, 4) == (0, 3)                                                   # {1, 2} gegen {0, 3}: die Seite mit dem Standort 0
    assert A._side_of(0b0111, 4) == (3,)                                                     # die kleinere Seite zuerst


def test_the_vector_in_a_degenerate_eigenspace_does_not_depend_on_the_numerical_basis():
    adj = adj_of(6, [(v, (v + 1) % 6) for v in range(6)])
    sp = A.spectrum(adj, "norm")
    assert sp.multiplicity == 2
    # drehe die Basis des Eigenraums: die Projektion des Hilfsvektors bleibt dieselbe
    vals, vecs = np.linalg.eigh(A.laplacian(adj, "norm"))
    sub = vecs[:, 1:3]
    theta = 0.7
    rot = sub @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    h = A.helper_vector(6)
    v1, v2 = sub @ (sub.T @ h), rot @ (rot.T @ h)
    assert np.allclose(v1, v2)
    assert np.allclose(np.array(sp.vec), v1 / np.linalg.norm(v1))


def test_make_instance_never_has_an_isolated_location():
    for kind in C.KINDS:
        for seed in range(1, 6):
            inst = S.make_instance(kind, C.N_MIN, C.DEG_MIN, C.LINKS_MIN, seed)
            assert min(len(a) for a in inst.adj) >= 1                                       # sonst teilt die normalisierte Matrix durch null


def test_validate_reports_every_kind_of_problem_on_tampered_analyses():
    a = ev.analyse(ev.Settings())
    assert ev.validate(a) == []
    norm = a.spectra["norm"]
    zero = dataclasses.replace(norm, lam2=0.0)
    assert any("nicht zusammenhängend" in p for p in ev.validate(dataclasses.replace(a, spectra={**a.spectra, "norm": zero})))
    bad_vec = dataclasses.replace(norm, vec=tuple(reversed(norm.vec)) if tuple(reversed(norm.vec)) != norm.vec else (1.0,) + (0.0,) * 7)
    assert any("Eigenwertgleichung" in p for p in ev.validate(dataclasses.replace(a, spectra={**a.spectra, "norm": bad_vec})))
    sw = a.sweeps["plain"]
    bad_sweep = dataclasses.replace(sw, cut=(sw.cut[0] + 1,) + sw.cut[1:])
    assert any("unnormalisiert" in p and "Nachzählung" in p for p in ev.validate(dataclasses.replace(a, sweeps={**a.sweeps, "plain": bad_sweep})))
    bad_cond = dataclasses.replace(a.sweeps["norm"], cond=(a.sweeps["norm"].cond[0] * 2,) + a.sweeps["norm"].cond[1:])
    assert any("normalisiert" in p and "Nachzählung" in p for p in ev.validate(dataclasses.replace(a, sweeps={**a.sweeps, "norm": bad_cond})))
    opts = dict(a.exact.opt)
    opts["cond"] = dataclasses.replace(opts["cond"], value=opts["cond"].value * 10)
    problems = ev.validate(dataclasses.replace(a, exact=dataclasses.replace(a.exact, opt=opts)))
    assert any("Cheeger" in p for p in problems) and any("Sweep-Cut liegt nicht" in p for p in problems) and any("gemeldeten Wert" in p for p in problems)
    assert any("Stoer–Wagner weicht" in p for p in ev.validate(dataclasses.replace(a, mincut=2)))
    assert any("kleinste Grad" in p for p in ev.validate(dataclasses.replace(a, mincut=99, exact=None)))
    lowered = dataclasses.replace(a.exact.opt["cond"], value=a.exact.opt["cond"].value * 0.1)
    below = dict(a.exact.opt)
    below["cond"] = lowered
    assert any("Cheeger" in p for p in ev.validate(dataclasses.replace(a, exact=dataclasses.replace(a.exact, opt=below))))               # h unter der unteren Schranke


def test_the_enumeration_limit_is_inclusive_and_the_analysis_is_deterministic():
    assert ev.analyse(ev.Settings(kind="random", n=C.BRUTE_N, seed=3)).exact is not None
    assert ev.analyse(ev.Settings(kind="random", n=C.BRUTE_N + 1, seed=3)).exact is None
    a1, a2 = ev.analyse(ev.Settings(kind="random", n=14, seed=5)), ev.analyse(ev.Settings(kind="random", n=14, seed=5))
    assert a1.sweeps == a2.sweeps and a1.spectra == a2.spectra and a1.exact == a2.exact


def test_the_analysis_properties_follow_the_settings():
    a = ev.analyse(ev.Settings(kind="random", n=14, seed=5, lap="plain", measure="ratio"))
    assert a.spec is a.spectra["plain"] and a.sw is a.sweeps["plain"] and a.best_k == a.sw.best("ratio") and a.median_k == 7
    assert a.side == a.sw.prefix(a.best_k) and a.sweep_value == a.sw.ratio[a.best_k - 1] and a.optimum == a.exact.opt["ratio"].value and a.h == a.exact.opt["cond"].value
    assert a.side_measures[0] == A.cut_size(a.inst.adj, a.side) and a.cheeger == A.cheeger_bounds(a.spectra["norm"].lam2) and a.m == a.inst.m and a.n == 14
    big = ev.analyse(ev.Settings(kind="random", n=40, seed=5))
    assert big.optimum is None and big.h is None


def test_size_boundaries_of_the_regler_range():
    for n in (C.N_MIN, C.N_MAX):
        a = ev.analyse(ev.Settings(kind="random", n=n, deg=C.DEG_MIN, links=C.LINKS_MIN, seed=1))
        assert a.n == n and ev.validate(a) == []
    with pytest.raises(ValueError):
        A.spectrum(adj_of(3, [(0, 1), (1, 2)]), "signless")
