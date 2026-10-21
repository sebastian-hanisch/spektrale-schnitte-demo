"""Experimente über feste Netze (nie ein einzelner Zufalls-Seed als Befund). Jede Funktion gibt eine Liste von Zeilen (dict) zurück; Verdikte und Abbildungen lesen nur diese Zeilen.

1. `exactness_rows`: stimmen Spektrum, Sweep-Cut, Cheeger-Schranken und Aufzählung miteinander überein (Eigenwertgleichung, Schranken, Nachzählung)?
2. `objective_rows`: wen wählen die Ziele? Kleinster Schnitt (trivial?), Ncut, RatioCut, Konduktanz; stimmen ihre besten Schnitte überein?
3. `cheeger_rows`: wie eng sind die Schranken von Cheeger?
4. `sweep_rows`: wie oft trifft der Sweep-Cut das Optimum, wie weit liegt er schlimmstenfalls darüber?
5. `big_rows`: im großen Netz: Sweep-Cut gegen die Halbierung am Median des Fiedler-Vektors und gegen den kleinsten Schnitt."""

import numpy as np

import spk_algorithm as A
import spk_constants as C
import spk_scenario as S


def kind_label(kind, deg):
    name = C.KIND_NAMES[kind]
    return f"{name}, Grad {deg:g}".replace(".", ",") if kind in ("random", "verbund") else name


def _net(kind, deg, n, seed):
    return S.make_instance(kind, n, deg, 1, seed)


def _close(a, b):
    return abs(a - b) <= C.TOL * max(1.0, abs(a), abs(b))


def _residual(adj, spec):
    lap = A.laplacian(adj, spec.kind)
    v = np.array(spec.vec)
    return float(np.linalg.norm(lap @ v - spec.lam2 * v))


def _sweep_recount_ok(adj, sw):
    for k in range(1, len(adj)):
        c, r, nc, co = A.measures(adj, sw.prefix(k))
        if not (c == sw.cut[k - 1] and _close(r, sw.ratio[k - 1]) and _close(nc, sw.ncut[k - 1]) and _close(co, sw.cond[k - 1])):
            return False
    return True


def exactness_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        checks = wrong = 0
        for seed in C.SWEEP_SEEDS:
            inst = _net(kind, deg, C.EX_N, seed)
            adj = inst.adj
            ex = A.exact_optima(adj)
            h = ex.opt["cond"].value
            res = []
            specs = {lap: A.spectrum(adj, lap) for lap in C.LAPS}
            sweeps = {lap: A.sweep(adj, specs[lap].x) for lap in C.LAPS}
            lo, hi = A.cheeger_bounds(specs["norm"].lam2)
            res += [lo <= h + C.TOL, h <= hi + C.TOL, h - C.TOL <= min(sweeps["norm"].cond) <= hi + C.TOL]
            degs = np.sqrt(np.array(A.degrees(adj), dtype=float))
            for lap in C.LAPS:
                sp, sw = specs[lap], sweeps[lap]
                res += [sp.lam2 > C.TOL, _residual(adj, sp) < 1e-8, _sweep_recount_ok(adj, sw)]
                trivial_vec = degs if lap == "norm" else np.ones(len(adj))
                res.append(abs(float(np.dot(trivial_vec, np.array(sp.vec)))) < 1e-8 * float(np.linalg.norm(trivial_vec)))
                for measure in ("cond", "ncut", "ratio"):
                    res.append(min(sw.series(measure)) >= ex.opt[measure].value - C.TOL)
            for key in C.OBJECTIVES:
                side = ex.opt[key].side
                c, r, nc, co = A.measures(adj, side)
                value = {"cut": c, "ratio": r, "ncut": nc, "cond": co}[key]
                res.append(_close(value, ex.opt[key].value))
            checks += len(res)
            wrong += res.count(False)
        rows.append({"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": len(C.SWEEP_SEEDS), "checks": checks, "wrong": wrong, "n": C.EX_N})
    return rows


def objective_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": C.EX_N, "cut_all_trivial": 0, "cut_any_trivial": 0, "ncut_trivial": 0, "ratio_trivial": 0, "cond_trivial": 0,
               "same_ncut_ratio": 0, "same_ncut_cond": 0, "sum_cuts": 0}
        for seed in C.SWEEP_SEEDS:
            ex = A.exact_optima(_net(kind, deg, C.EX_N, seed).adj)
            row["nets"] += 1
            row["cut_all_trivial"] += ex.opt["cut"].all_trivial
            row["cut_any_trivial"] += ex.opt["cut"].any_trivial
            row["ncut_trivial"] += ex.opt["ncut"].all_trivial
            row["ratio_trivial"] += ex.opt["ratio"].all_trivial
            row["cond_trivial"] += ex.opt["cond"].all_trivial
            row["same_ncut_ratio"] += ex.same("ncut", "ratio")
            row["same_ncut_cond"] += ex.same("ncut", "cond")
            row["sum_cuts"] += ex.opt["cut"].count
        rows.append(row)
    return rows


def cheeger_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": C.EX_N, "violations": 0, "lo_min": None, "lo_max": None, "hi_min": None, "hi_max": None}
        for seed in C.SWEEP_SEEDS:
            adj = _net(kind, deg, C.EX_N, seed).adj
            h = A.exact_optima(adj).opt["cond"].value
            lam2 = A.spectrum(adj, "norm").lam2
            lo, hi = A.cheeger_bounds(lam2)
            row["nets"] += 1
            row["violations"] += (h < lo - C.TOL) + (h > hi + C.TOL)
            lo_r, hi_r = h / lo, h / hi
            row["lo_min"] = lo_r if row["lo_min"] is None else min(row["lo_min"], lo_r)
            row["lo_max"] = lo_r if row["lo_max"] is None else max(row["lo_max"], lo_r)
            row["hi_min"] = hi_r if row["hi_min"] is None else min(row["hi_min"], hi_r)
            row["hi_max"] = hi_r if row["hi_max"] is None else max(row["hi_max"], hi_r)
        rows.append(row)
    return rows


def sweep_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": C.EX_N, "hit_cond": 0, "hit_cond_plain": 0, "hit_ncut": 0, "hit_ratio_plain": 0, "worst_cond": 1.0, "sum_cond_ratio": 0.0,
               "multiple": 0}
        for seed in C.SWEEP_SEEDS:
            adj = _net(kind, deg, C.EX_N, seed).adj
            ex = A.exact_optima(adj)
            norm, plain = A.spectrum(adj, "norm"), A.spectrum(adj, "plain")
            sn, sp = A.sweep(adj, norm.x), A.sweep(adj, plain.x)
            row["nets"] += 1
            row["multiple"] += norm.multiplicity > 1
            h = ex.opt["cond"].value
            row["hit_cond"] += _close(min(sn.cond), h)
            row["hit_cond_plain"] += _close(min(sp.cond), h)
            row["hit_ncut"] += _close(min(sn.ncut), ex.opt["ncut"].value)
            row["hit_ratio_plain"] += _close(min(sp.ratio), ex.opt["ratio"].value)
            ratio = min(sn.cond) / h
            row["worst_cond"] = max(row["worst_cond"], ratio)
            row["sum_cond_ratio"] += ratio
        rows.append(row)
    return rows


def big_rows():
    rows = []
    for kind, deg in C.EX_KINDS:
        row = {"kind": kind, "deg": deg, "label": kind_label(kind, deg), "nets": 0, "n": 0, "sum_phi_sweep": 0.0, "sum_phi_median": 0.0, "sum_cut_sweep": 0, "sum_cut_median": 0, "sum_small": 0.0,
               "sum_mincut": 0, "sum_lam2": 0.0, "better": 0}
        for seed in C.SWEEP_SEEDS[:C.EX_BIG_SEEDS]:
            inst = _net(kind, deg, C.EX_BIG_N, seed)
            adj = inst.adj
            n = inst.n
            sp = A.spectrum(adj, "norm")
            sw = A.sweep(adj, sp.x)
            k = sw.best("cond")
            half = n // 2
            row["nets"] += 1
            row["n"] = n
            row["sum_phi_sweep"] += sw.cond[k - 1]
            row["sum_phi_median"] += sw.cond[half - 1]
            row["sum_cut_sweep"] += sw.cut[k - 1]
            row["sum_cut_median"] += sw.cut[half - 1]
            row["sum_small"] += min(k, n - k) / n
            row["sum_mincut"] += A.min_cut(adj)[0]
            row["sum_lam2"] += sp.lam2
            row["better"] += sw.cond[k - 1] < sw.cond[half - 1] - C.TOL
        rows.append(row)
    return rows
