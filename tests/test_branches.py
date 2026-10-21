"""Zweige der Hauptansicht (Nullspalte-Lehre): Jeder Weg, den die App zeigt, kommt in mindestens einem Preset oder einem festen Netz vor.

Ein Preset, das seine Beschreibung nicht einlöst, fällt in `test_claims.py` auf. Hier: die Zweige und dass jede Abbildung für jeden Zweig gebaut werden kann (kein Fehler, nichtleere Daten)."""

import spk_constants as C
import spk_evaluation as ev
import spk_visualization as V

from conftest import cached


def _all():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_every_preset_validates_clean():
    for name, a in _all().items():
        assert ev.validate(a) == [], name


def test_every_branch_of_the_question_is_reached():
    analyses = list(_all().values())
    assert any(a.exact is None for a in analyses) and any(a.exact is not None for a in analyses)
    assert any(a.mincut_trivial for a in analyses) and any(not a.mincut_trivial for a in analyses)
    assert any(a.exact and a.exact.opt["cut"].count > 1 for a in analyses) and any(a.exact and a.exact.opt["cut"].count == 1 for a in analyses)
    assert any(a.exact and abs(a.sweep_value - a.optimum) > C.TOL for a in analyses) and any(a.exact and abs(a.sweep_value - a.optimum) <= C.TOL for a in analyses)
    assert any(a.spec.multiplicity > 1 for a in analyses) and any(a.spec.multiplicity == 1 for a in analyses)
    assert {a.settings.measure for a in analyses} == {"cond", "ncut"} and {a.settings.lap for a in analyses} == {"norm"}


def test_the_plain_matrix_and_the_ratio_measure_are_reachable_through_the_regler():
    a = ev.analyse(ev.Settings(**{**C.PRESETS["Gitter: λ₂ ist doppelt"], "lap": "plain", "measure": "ratio"}))
    assert a.spec is a.spectra["plain"] and a.sweep_value == a.sw.ratio[a.best_k - 1] and ev.validate(a) == []
    assert ev.verdict_question(a)[1].count("RatioCut") >= 1


def test_every_net_kind_is_used_by_a_preset_or_a_known_extra():
    kinds = {p["kind"] for p in C.PRESETS.values()}
    assert kinds >= {"textbook", "random", "grid", "scalefree", "verbund", "geometric"}


def test_the_figures_build_for_every_preset():
    for name, a in _all().items():
        inst = a.inst
        assert V.build_fiedler_map(inst, a.spec.x, a.side).data and V.build_partition_map(inst, a.side).data, name
        assert V.build_sorted_values(a).data and V.build_sweep_curve(a).data, name
        assert V.build_cheeger(a).data, name
        for lap in C.LAPS:
            for measure in C.MEASURES:
                b = ev.analyse(ev.Settings(**{**C.PRESETS[name], "lap": lap, "measure": measure})) if inst.n <= 30 else a
                assert V.build_sweep_curve(b).data
        if a.exact:
            for key in C.OBJECTIVES:
                assert V.build_partition_map(inst, a.exact.opt[key].side, height=340).data


def test_the_experiment_figures_build_from_the_cached_rows():
    for fig, rows in ((V.build_exact, "exactness_rows"), (V.build_objective, "objective_rows"), (V.build_cheeger_rows, "cheeger_rows"), (V.build_sweep_hits, "sweep_rows"), (V.build_big, "big_rows")):
        assert fig(cached(rows)).data
