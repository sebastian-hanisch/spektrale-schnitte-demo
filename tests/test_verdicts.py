"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds; so prüft der Test Grenzen und jede genannte Zahl direkt."""

import dataclasses

import spk_constants as C
import spk_evaluation as ev


def test_formats():
    assert ev._i(1234567) == "1.234.567" and ev._i(7) == "7"
    assert ev._dec(3.0) == "3" and ev._dec(1234.5) == "1.234,5" and ev._dec(0.75) == "0,8" and ev._dec(2.25, 2) == "2,25"
    assert ev._v(0.14285714) == "0,143" and ev._v(1.0) == "1" and ev._v(0.2) == "0,200"
    assert ev._pct(0.256) == "26" and ev._pct(0.256, 1) == "25,6"
    assert ev._of(3, 1200) == "3 von 1.200" and ev._leitungen(1) == "1 Leitung" and ev._leitungen(2) == "2 Leitungen" and ev._leitungen(1000) == "1.000 Leitungen"


def test_verdict_exact_counts_and_warns_on_any_deviation():
    rows = [{"nets": 3, "checks": 30, "wrong": 0, "n": 8}, {"nets": 2, "checks": 2000, "wrong": 0, "n": 8}]
    kind, text = ev.verdict_exact(rows)
    assert kind == "info" and text.startswith("Auf 5 Netzen mit 8 Standorten (2 Netztypen) bestehen 2.030 Prüfungen:") and text.endswith("Abweichungen: 0.")
    rows[1]["wrong"] = 1
    rows[0]["wrong"] = 2
    assert ev.verdict_exact(rows) == ("warning", "Auf 5 Netzen weichen 3 von 2.030 Prüfungen ab.")


def test_verdict_objective_counts_every_objective():
    rows = [{"nets": 2, "cut_all_trivial": 2, "cut_any_trivial": 2, "ncut_trivial": 0, "ratio_trivial": 1, "cond_trivial": 0, "same_ncut_ratio": 1, "same_ncut_cond": 2},
            {"nets": 2, "cut_all_trivial": 0, "cut_any_trivial": 1, "ncut_trivial": 0, "ratio_trivial": 0, "cond_trivial": 0, "same_ncut_ratio": 2, "same_ncut_cond": 0}]
    kind, text = ev.verdict_objective(rows)
    assert kind == "info"
    assert text == ("Der kleinste Schnitt besteht in 2 von 4 Netzen nur aus einzelnen Standorten (in 3 von 4 mindestens einer von ihnen); der beste Ncut-Schnitt ist es in 0 von 4 Netzen, der beste RatioCut-Schnitt in "
                    "1 von 4 und der mit der kleinsten Konduktanz in 0 von 4. Ncut und RatioCut haben dieselben besten Schnitte in 3 von 4 Netzen, Ncut und Konduktanz in 2 von 4.")


def test_verdict_cheeger_ranges_and_warning():
    rows = [{"nets": 2, "violations": 0, "lo_min": 1.5, "lo_max": 2.5, "hi_min": 0.2, "hi_max": 0.3}, {"nets": 2, "violations": 0, "lo_min": 1.3, "lo_max": 3.0, "hi_min": 0.15, "hi_max": 0.4}]
    kind, text = ev.verdict_cheeger(rows)
    assert kind == "info"
    assert text == ("Die Schranken von Cheeger gelten in allen 4 Netzen (Verletzungen: 0). Die kleinste Konduktanz h liegt das 1,3- bis 3-Fache über der unteren Schranke λ₂/2 und bei 15 bis 40 % der oberen "
                    "Schranke √(2 λ₂): keine der beiden ist in diesen Netzen eng.")
    rows[0]["violations"] = 1
    assert ev.verdict_cheeger(rows) == ("warning", "Die Schranken von Cheeger sind in 1 Fall verletzt.")
    rows[1]["violations"] = 1
    assert ev.verdict_cheeger(rows) == ("warning", "Die Schranken von Cheeger sind in 2 Fällen verletzt.")


def test_verdict_sweep_counts_hits_and_names_degenerate_eigenvalues():
    rows = [{"nets": 2, "hit_cond": 1, "hit_cond_plain": 2, "hit_ncut": 1, "hit_ratio_plain": 2, "worst_cond": 1.25, "multiple": 0},
            {"nets": 2, "hit_cond": 0, "hit_cond_plain": 1, "hit_ncut": 2, "hit_ratio_plain": 0, "worst_cond": 1.5, "multiple": 2}]
    kind, text = ev.verdict_sweep(rows)
    assert kind == "info"
    assert text == ("Der Sweep-Cut trifft die kleinste Konduktanz in 1 von 4 Netzen mit der normalisierten Matrix und in 3 von 4 mit der unnormalisierten, den kleinsten Ncut (normalisiert) in 3 von 4 und den "
                    "kleinsten RatioCut (unnormalisiert) in 2 von 4; schlimmstenfalls liegt seine Konduktanz das 1,5-Fache über h. In 2 von 4 Netzen ist λ₂ mehrfach (Gitter): dort hängt der Sweep von der "
                    "Wahl des Fiedler-Vektors im Eigenraum ab.")
    rows[1]["multiple"] = 0
    assert "mehrfach" not in ev.verdict_sweep(rows)[1] and ev.verdict_sweep(rows)[1].endswith("über h.")


def test_verdict_big_gives_the_means_per_net_and_names_the_largest_difference():
    rows = [{"label": "A", "nets": 2, "n": 20, "sum_phi_sweep": 0.2, "sum_phi_median": 0.6, "better": 1, "sum_small": 0.8},
            {"label": "B", "nets": 2, "n": 20, "sum_phi_sweep": 0.4, "sum_phi_median": 0.4, "better": 0, "sum_small": 1.0}]
    kind, text = ev.verdict_big(rows)
    assert kind == "info"
    assert text == ("Bei 20 Standorten (4 Netze) liegt die Konduktanz des besten Sweep-Anfangs im Mittel bei 0,150, die der Halbierung am Median des Fiedler-Vektors bei 0,250; in 1 von 4 Netzen ist der beste "
                    "Anfang nicht die Hälfte. Die kleinere Seite hat im Mittel 45 % der Standorte. Der größte Unterschied: A (die Halbierung hat im Mittel das 3,0-Fache der Konduktanz des Sweeps).")
    rows[1]["sum_phi_sweep"] = 0.05
    assert "Der größte Unterschied: B (die Halbierung hat im Mittel das 8-Fache" in ev.verdict_big(rows)[1]


def test_verdict_question_for_the_textbook_network_by_hand():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_question(a)
    assert kind == "info"
    assert text == ("**Bei 8 Standorten kostet der kleinste Schnitt 1; 2 Schnitte haben diesen Wert, einer davon trennt nur einen einzelnen Standort ab; der Sweep-Cut mit Konduktanz wählt dagegen einen Schnitt mit "
                    "1 Leitung und 5 gegen 3 Standorten.** Konduktanz = 0,143, das Optimum der Aufzählung. Cheeger: λ₂ = 0,149 (normalisiert), also liegt die kleinste Konduktanz h zwischen 0,074 und 0,545; sie ist 0,143.")


def test_verdict_question_branches():
    def q(name, **over):
        return ev.verdict_question(ev.analyse(ev.Settings(**{**C.PRESETS[name], **over})))[1]
    miss = q("Der Sweep verfehlt das Optimum")
    assert "wählt dagegen einen Schnitt mit 7 Leitungen und 9 gegen 7 Standorten" in miss and "Konduktanz = 0,333, das 1,22-Fache des Optimums (0,273)." in miss
    assert "3 Schnitte haben diesen Wert, alle trennen nur einen einzelnen Standort ab" in miss
    single = q("Cheeger ist nicht eng")
    assert "kostet der kleinste Schnitt 1, er trennt 4 von 16 Standorten ab;" in single and "Schnitte haben" not in single
    grid = q("Gitter: λ₂ ist doppelt")
    assert grid.endswith("λ₂ ist 2-fach: der Fiedler-Vektor ist nicht eindeutig.") and "4 Schnitte haben diesen Wert, alle trennen" in grid
    big = q("Großes Kartennetz")
    assert "er trennt 8 von 100 Standorten ab" in big and "1 Leitung und 59 gegen 41 Standorten" in big and "(ohne Aufzählung keine Gegenprobe)" in big and "sie ist" not in big
    verbund = q("Der Sweep findet einen Verbund")
    assert "er trennt nur einen einzelnen Standort ab;" in verbund and "2 Leitungen und 30 gegen 60 Standorten" in verbund
    assert "Ncut = " in q("Ncut teilt, der kleinste Schnitt nicht") and "Sweep-Cut mit Ncut" in q("Ncut teilt, der kleinste Schnitt nicht")
    some = q("Lehrbeispiel (Hantel mit Anhängsel)", seed=1)
    assert some == q("Lehrbeispiel (Hantel mit Anhängsel)")                                              # das Lehrbeispiel ignoriert den Zufalls-Seed
    a = ev.analyse(ev.Settings(**C.PRESETS["Cheeger ist nicht eng"]))
    none = dataclasses.replace(a, exact=None)
    assert "ohne Aufzählung keine Gegenprobe" in ev.verdict_question(none)[1] and "kleinste Schnitt 1, er trennt" in ev.verdict_question(none)[1]
