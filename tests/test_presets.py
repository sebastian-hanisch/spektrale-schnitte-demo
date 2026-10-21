"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen; Permalink-Grenzen."""

import spk_constants as C
import spk_evaluation as ev
import spk_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        assert preset["kind"] in C.KINDS and preset["lap"] in C.LAPS and preset["measure"] in C.MEASURES, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.DEG_MIN <= preset["deg"] <= C.DEG_MAX and C.LINKS_MIN <= preset["links"] <= C.LINKS_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_the_first_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[next(iter(C.PRESETS))]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_every_choice_field_is_listed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert {spec.url_param for spec in P.SETTING_SPECS.values()} == {"kind", "n", "deg", "links", "seed", "lap", "measure"}
    assert set(P.CHOICES) == {"kind_select", "lap_select", "measure_select"} and P.CHOICES["lap_select"] == C.LAPS and P.CHOICES["measure_select"] == C.MEASURES


def test_every_kind_has_a_name_a_help_text_and_a_regler_note_and_every_choice_a_name():
    assert set(C.KIND_NAMES) == set(C.KIND_HELP) == set(C.KIND_REGLER) == set(C.KINDS)
    assert set(C.LAP_NAMES) == set(C.LAPS) and set(C.MEASURE_NAMES) == set(C.MEASURE_SHORT) == set(C.MEASURES) and set(C.OBJECTIVE_NAMES) == set(C.OBJECTIVES)
    assert C.DEFAULT_LAP in C.LAPS and C.DEFAULT_MEASURE in C.MEASURES


def test_the_experiment_grid_uses_only_known_kinds_and_sizes_inside_the_regler_range():
    assert all(kind in C.KINDS and kind != "textbook" for kind, _deg in C.EX_KINDS)
    for n in (C.EX_N, C.EX_BIG_N):
        assert C.N_MIN <= n <= C.N_MAX and int(n ** 0.5) ** 2 == n                      # Quadratzahl: das Gitter hat dieselbe Größe wie die anderen Netztypen
    assert C.EX_N <= C.BRUTE_N < C.EX_BIG_N and C.EX_BIG_SEEDS <= len(C.SWEEP_SEEDS) and 0 < C.TOL < 1e-6


def test_the_presets_that_use_the_enumeration_stay_within_its_limit():
    for name, preset in C.PRESETS.items():
        n = ev.analyse(ev.Settings(**preset)).n
        assert (n <= C.BRUTE_N) == (name not in ("Der Sweep findet einen Verbund", "Großes Kartennetz")), name
