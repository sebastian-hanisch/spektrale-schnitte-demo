"""Jede Zahl im README (Abschnitt „Befunde“) wird hier über dieselben Experimentfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zähler und Mittel über feste Netze (je Netztyp zwölf Netze mit 16 Standorten, feste Zufalls-Seeds; im großen Netz vier je Typ mit 100 Standorten), nie ein einzelner Zufalls-Seed. Die Zahlen der Vormessung im Einzelplan
stammen aus Wegwerfskripten mit anderen Netzen; maßgeblich sind diese hier."""

import pytest

import spk_constants as C
import spk_evaluation as ev

from conftest import cached

LABELS = ["Zufallsnetz, Grad 3", "Zufallsnetz, Grad 5", "Gitter", "Ring mit Abkürzungen", "Skalenfreies Netz", "Verbundnetz, Grad 5", "Kartennetz"]


def test_spectrum_sweep_cheeger_and_enumeration_agree_on_every_small_network():
    rows = cached("exactness_rows")
    assert [r["label"] for r in rows] == LABELS
    assert len(rows) == 7 and sum(r["nets"] for r in rows) == 84 and sum(r["checks"] for r in rows) == 1764 and sum(r["wrong"] for r in rows) == 0
    assert all(r["n"] == 16 and r["checks"] == 252 for r in rows)                              # je Netz 21 Prüfungen


def test_the_smallest_cut_is_a_single_location_but_the_balanced_measures_are_not():
    rows = cached("objective_rows")
    assert [r["cut_all_trivial"] for r in rows] == [10, 12, 12, 12, 12, 0, 1] and sum(r["cut_all_trivial"] for r in rows) == 59
    assert [r["cut_any_trivial"] for r in rows] == [12, 12, 12, 12, 12, 0, 5] and sum(r["cut_any_trivial"] for r in rows) == 65
    assert [r["ncut_trivial"] for r in rows] == [0] * 7 and [r["cond_trivial"] for r in rows] == [0] * 7
    assert [r["ratio_trivial"] for r in rows] == [2, 8, 0, 0, 0, 0, 0] and sum(r["ratio_trivial"] for r in rows) == 10
    assert [r["sum_cuts"] for r in rows] == [28, 19, 48, 40, 88, 36, 62]                       # Zahl aller kleinsten Schnitte, über je zwölf Netze
    assert all(r["nets"] == 12 for r in rows) and sum(r["nets"] for r in rows) == 84


def test_ncut_ratiocut_and_conductance_choose_different_cuts():
    rows = cached("objective_rows")
    assert [r["same_ncut_ratio"] for r in rows] == [8, 0, 12, 9, 5, 12, 12] and sum(r["same_ncut_ratio"] for r in rows) == 58
    assert [r["same_ncut_cond"] for r in rows] == [4, 4, 12, 6, 2, 12, 10] and sum(r["same_ncut_cond"] for r in rows) == 50


def test_the_bounds_of_cheeger_hold_and_are_never_tight():
    rows = cached("cheeger_rows")
    assert all(r["violations"] == 0 and r["nets"] == 12 for r in rows)
    lo_min, lo_max = min(r["lo_min"] for r in rows), max(r["lo_max"] for r in rows)
    hi_min, hi_max = min(r["hi_min"] for r in rows), max(r["hi_max"] for r in rows)
    assert (round(lo_min, 2), round(lo_max, 2)) == (1.28, 3.08) and (round(hi_min, 2), round(hi_max, 2)) == (0.15, 0.39)
    assert [round(r["lo_min"], 2) for r in rows] == [1.59, 1.28, 1.53, 1.49, 1.51, 1.47, 1.68] and [round(r["hi_max"], 2) for r in rows] == [0.37, 0.38, 0.25, 0.35, 0.39, 0.16, 0.32]


def test_how_often_the_sweep_cut_hits_the_optimum():
    rows = cached("sweep_rows")
    assert [r["hit_cond"] for r in rows] == [8, 4, 0, 5, 7, 12, 12] and sum(r["hit_cond"] for r in rows) == 48
    assert [r["hit_cond_plain"] for r in rows] == [8, 1, 12, 7, 7, 12, 12] and sum(r["hit_cond_plain"] for r in rows) == 59
    assert [r["hit_ncut"] for r in rows] == [9, 6, 0, 8, 11, 12, 12] and sum(r["hit_ncut"] for r in rows) == 58
    assert [r["hit_ratio_plain"] for r in rows] == [11, 12, 12, 5, 10, 12, 12] and sum(r["hit_ratio_plain"] for r in rows) == 74
    no_grid = [r for r in rows if r["label"] != "Gitter"]
    assert len(no_grid) == 6 and sum(r["hit_cond"] for r in no_grid) == 48 and sum(r["hit_cond_plain"] for r in no_grid) == 47     # ohne das (zwölfmal gleiche) Gitter kein Unterschied der Matrizen
    assert round(max(r["worst_cond"] for r in rows), 2) == 1.43 and [round(r["worst_cond"], 2) for r in rows] == [1.22, 1.19, 1.43, 1.2, 1.17, 1.0, 1.0]


def test_on_the_grid_lambda2_is_always_double_and_the_sweep_depends_on_the_choice_in_the_eigenspace():
    rows = cached("sweep_rows")
    assert [r["multiple"] for r in rows] == [0, 0, 12, 0, 0, 0, 0]
    grid = rows[2]
    assert grid["hit_cond"] == 0 and grid["hit_ncut"] == 0 and grid["hit_cond_plain"] == 12 and grid["hit_ratio_plain"] == 12          # mit der Wahl der Demo und mit der unnormalisierten Matrix


def test_in_the_large_network_the_best_sweep_start_beats_the_median_halving():
    rows = cached("big_rows")
    assert all(r["n"] == 100 and r["nets"] == 4 for r in rows)
    phi_sweep = [r["sum_phi_sweep"] for r in rows]
    phi_median = [r["sum_phi_median"] for r in rows]
    assert phi_sweep == pytest.approx([0.5847, 0.9424, 0.3111, 0.2126, 0.8402, 0.0475, 0.0463], abs=5e-5)
    assert phi_median == pytest.approx([0.598, 0.9773, 0.3111, 0.3332, 0.8466, 0.4914, 0.1776], abs=5e-5)
    assert [r["better"] for r in rows] == [1, 3, 0, 4, 2, 4, 4] and sum(r["better"] for r in rows) == 18
    assert [r["sum_cut_sweep"] for r in rows] == [82, 225, 56, 21, 162, 8, 6] and [r["sum_cut_median"] for r in rows] == [88, 236, 56, 66, 162, 121, 33]
    assert [r["sum_mincut"] for r in rows] == [4, 5, 8, 11, 8, 5, 4]
    assert [r["sum_small"] for r in rows] == pytest.approx([1.93, 1.94, 2.0, 1.05, 1.97, 1.33, 1.43], abs=5e-3) and round(sum(phi_sweep) / 28, 3) == 0.107 and round(sum(phi_median) / 28, 3) == 0.133
    assert round(rows[5]["sum_phi_median"] / rows[5]["sum_phi_sweep"], 1) == 10.3 and round(rows[6]["sum_phi_median"] / rows[6]["sum_phi_sweep"], 1) == 3.8


def test_the_preset_descriptions_hold():
    def a(name):
        return ev.analyse(ev.Settings(**C.PRESETS[name]))
    lehr = a("Lehrbeispiel (Hantel mit Anhängsel)")
    assert (lehr.mincut, lehr.exact.opt["cut"].count, lehr.exact.total) == (1, 2, 127) and lehr.exact.opt["ncut"].side == (4, 5, 6) and lehr.side == (0, 1, 2, 3, 7) and lehr.sweep_value == pytest.approx(1 / 7) and lehr.sweeps["plain"].prefix(lehr.sweeps["plain"].best("cond")) == (0, 1, 2, 3, 7)
    ncut = a("Ncut teilt, der kleinste Schnitt nicht")
    assert ncut.exact.opt["cut"].all_trivial and ncut.mincut == 2 and len(ncut.exact.opt["ncut"].side) == 7 and ncut.n - len(ncut.exact.opt["ncut"].side) == 9
    diff = a("Ncut und RatioCut wählen Verschiedenes")
    assert not diff.exact.same("ncut", "ratio") and (len(diff.exact.opt["ncut"].side), len(diff.exact.opt["ratio"].side)) == (4, 3)
    miss = a("Der Sweep verfehlt das Optimum")
    assert round(miss.sweep_value / miss.optimum, 1) == 1.2 and miss.sweep_value < miss.cheeger[1]
    grid = a("Gitter: λ₂ ist doppelt")
    assert grid.spec.multiplicity == 2 and round(grid.sweep_value / grid.optimum, 1) == 1.4
    loose = a("Cheeger ist nicht eng")
    assert loose.h / loose.cheeger[0] > 4 and round(loose.h / loose.cheeger[1], 2) == 0.29
    verbund = a("Der Sweep findet einen Verbund")
    sw = verbund.sw
    assert verbund.n == 90 and verbund.best_k == 30 and sw.cut[29] == 2 and round(sw.cond[29], 3) == 0.013 and round(sw.cond[44], 3) == 0.115
    big = a("Großes Kartennetz")
    assert big.n == 100 and big.exact is None and big.sw.cut[big.best_k - 1] == 1 and (big.best_k, big.n - big.best_k) == (59, 41) and big.mincut == 1 and not big.mincut_trivial
