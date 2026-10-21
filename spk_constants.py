"""Konstanten: Regler-Grenzen, feste Annahmen, Experiment-Raster, Farben und Presets."""

KINDS = ("textbook", "random", "grid", "ring", "scalefree", "verbund", "geometric")
KIND_NAMES = {"textbook": "Lehrbeispiel (Hantel mit Anhängsel)", "random": "Zufallsnetz", "grid": "Gitter", "ring": "Ring mit Abkürzungen", "scalefree": "Skalenfreies Netz", "verbund": "Verbundnetz",
              "geometric": "Kartennetz"}
KIND_HELP = {
    "textbook": "Acht Standorte, von Hand zu rechnen: zwei Werke, ein Verbindungsstück und ein Standort am Rand.",
    "random": "Zufallsbaum plus zufällige Leitungen; der mittlere Grad ist einstellbar.",
    "grid": "Standorte auf einem Raster, Verbindungen zu den vier Nachbarn.",
    "ring": "Vier nächste Nachbarn auf einem Ring, ein Zehntel der Leitungen führt zu einem zufälligen Standort.",
    "scalefree": "Neue Standorte hängen sich bevorzugt an gut vernetzte an (je zwei Leitungen).",
    "verbund": "Drei Regionen mit dichten inneren Verbindungen und wenigen Leitungen dazwischen (einstellbar).",
    "geometric": "Zufällige Punkte in einem Quadrat, jeder mit seinen drei nächsten Nachbarn verbunden: ein Straßennetz im Kleinen.",
}
KIND_REGLER = {
    "textbook": "Das Lehrbeispiel ist fest: Größe, Grad, Leitungen und Zufalls-Seed des Netzes wirken hier nicht (Matrix und Maß schon).",
    "random": "Wirksam: Standorte, mittlerer Grad, Zufalls-Seed.",
    "grid": "Wirksam: Standorte (gerundet auf ein Quadrat). Grad, Leitungen und Zufalls-Seed des Netzes wirken nicht.",
    "ring": "Wirksam: Standorte, Zufalls-Seed.",
    "scalefree": "Wirksam: Standorte, Zufalls-Seed.",
    "verbund": "Wirksam: Standorte, mittlerer Grad (in den Regionen), Leitungen zwischen den Regionen, Zufalls-Seed.",
    "geometric": "Wirksam: Standorte, Zufalls-Seed.",
}

# Matrix des Eigenvektors und Maß der Schnittwahl
LAPS = ("norm", "plain")
LAP_NAMES = {"norm": "normalisiert (L = I − D⁻½ A D⁻½)", "plain": "unnormalisiert (L = D − A)"}
MEASURES = ("cond", "ncut", "ratio")
MEASURE_NAMES = {"cond": "Konduktanz", "ncut": "Ncut (normalisierter Schnitt)", "ratio": "RatioCut"}
MEASURE_SHORT = {"cond": "Konduktanz", "ncut": "Ncut", "ratio": "RatioCut"}
DEFAULT_LAP, DEFAULT_MEASURE = "norm", "cond"

N_MIN, N_MAX, DEFAULT_N = 8, 120, 24
DEG_MIN, DEG_MAX, DEFAULT_DEG = 2.0, 10.0, 4.0
LINKS_MIN, LINKS_MAX, DEFAULT_LINKS = 1, 8, 2
SEED_MAX, DEFAULT_SEED = 999999, 7
DEFAULT_KIND = "textbook"
REGIONS = 3
RING_NEIGHBOURS, RING_REWIRE = 4, 0.1
SCALEFREE_M = 2
BRUTE_N = 18                             # größte Netzgröße für die Aufzählung aller Schnitte (App, Test, Experimente)
TOL = 1e-9                               # Gleichheit von Brüchen und Eigenwerten (Entartung, Gleichstände)
SIGNATURE = (0.3141592653, 1.2345678901) # feste Zahlen für den Hilfsvektor, der in einem entarteten Eigenraum den Vektor festlegt (plattformstabil)

SWEEP_SEEDS = tuple(range(100000, 100012))   # feste Netze für alle Experimente: nie ein einzelner Zufalls-Seed als Befund
EX_N = 16
EX_KINDS = (("random", 3.0), ("random", 5.0), ("grid", 0.0), ("ring", 0.0), ("scalefree", 0.0), ("verbund", 5.0), ("geometric", 0.0))
EX_BIG_N = 100                           # Quadratzahl: das Gitter hat dieselbe Größe wie die anderen Netztypen
EX_BIG_SEEDS = 4

COLOR_SIDE_A = "#2F6B65"
COLOR_SIDE_B = "#4c78a8"
COLOR_CUT = "#d1495b"
COLOR_NODE = "#c9d1d3"
COLOR_EDGE = "rgba(150,150,150,0.45)"
COLOR_BAR = "#4c78a8"
COLOR_BAR2 = "#e8a33d"
COLOR_BAR3 = "#2F6B65"
COLOR_LOCK = "#8d6cab"
OBJECTIVES = ("cut", "ratio", "ncut", "cond")
OBJECTIVE_NAMES = {"cut": "Kleinster Schnitt (ohne Balance)", "ratio": "RatioCut", "ncut": "Ncut", "cond": "Konduktanz (Cheeger-Konstante)"}

PRESETS = {
    "Lehrbeispiel (Hantel mit Anhängsel)": {"kind": "textbook", "n": DEFAULT_N, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": DEFAULT_SEED, "lap": DEFAULT_LAP, "measure": DEFAULT_MEASURE},
    "Ncut teilt, der kleinste Schnitt nicht": {"kind": "random", "n": 16, "deg": 5.0, "links": DEFAULT_LINKS, "seed": 1, "lap": "norm", "measure": "ncut"},
    "Ncut und RatioCut wählen Verschiedenes": {"kind": "scalefree", "n": 16, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 1, "lap": "norm", "measure": "ncut"},
    "Der Sweep verfehlt das Optimum": {"kind": "random", "n": 16, "deg": 3.0, "links": DEFAULT_LINKS, "seed": 17, "lap": "norm", "measure": "cond"},
    "Gitter: λ₂ ist doppelt": {"kind": "grid", "n": 16, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": DEFAULT_SEED, "lap": "norm", "measure": "cond"},
    "Cheeger ist nicht eng": {"kind": "geometric", "n": 16, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 26, "lap": "norm", "measure": "cond"},
    "Der Sweep findet einen Verbund": {"kind": "verbund", "n": 90, "deg": 5.0, "links": 1, "seed": 3, "lap": "norm", "measure": "cond"},
    "Großes Kartennetz": {"kind": "geometric", "n": 100, "deg": DEFAULT_DEG, "links": DEFAULT_LINKS, "seed": 5, "lap": "norm", "measure": "cond"},
}
PRESET_HELP = {
    "Lehrbeispiel (Hantel mit Anhängsel)": "Acht Standorte, von Hand zu prüfen: Der kleinste Schnitt kostet 1 (ein Standort allein oder das Verbindungsstück), Ncut, RatioCut und Konduktanz trennen die beiden Werke.",
    "Ncut teilt, der kleinste Schnitt nicht": "Der kleinste Schnitt trennt nur einen einzelnen Standort ab, der beste Ncut-Schnitt teilt das Netz in sieben und neun Standorte.",
    "Ncut und RatioCut wählen Verschiedenes": "Die besten Schnitte nach Ncut und nach RatioCut sind verschieden: vier Standorte gegen zwölf, drei gegen dreizehn.",
    "Der Sweep verfehlt das Optimum": "Der beste Anfang der Sortierung nach dem Fiedler-Vektor hat eine größere Konduktanz als der beste Schnitt der Aufzählung (das 1,2-Fache), bleibt aber unter √(2 λ₂).",
    "Gitter: λ₂ ist doppelt": "Das quadratische Gitter hat einen doppelten Eigenwert λ₂: Der Fiedler-Vektor ist nicht eindeutig, die Wahl im Eigenraum entscheidet über den Sweep (mit der Wahl der Demo: normalisiert 1,4-fach über h, unnormalisiert trifft h; bei anderer Wahl kann es umgekehrt sein).",
    "Cheeger ist nicht eng": "Die kleinste Konduktanz h liegt mehr als das Vierfache über der unteren Schranke λ₂ / 2 und bei etwa 29 % der oberen: Die Schranken gelten, sagen aber wenig.",
    "Der Sweep findet einen Verbund": "Im Verbundnetz mit 90 Standorten schneidet der Sweep genau eine der drei Regionen ab (30 Standorte, 2 Leitungen); die Halbierung am Median hat eine Konduktanz von 0,115 statt 0,013.",
    "Großes Kartennetz": "100 Standorte, keine Aufzählung mehr: Der Sweep braucht eine einzige Eigenzerlegung und findet einen sehr dünnen Schnitt mit einer Leitung.",
}
