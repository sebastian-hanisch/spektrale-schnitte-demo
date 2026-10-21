# Spektrale Schnitte: Fiedler-Vektor, Sweep-Cut und Cheeger – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-spektrale-schnitte-demo.streamlit.app/)** -->

Siebtes Stück der Reihe **Schnitte und Zusammenhang** (Arbeitstitel der Linie `schnitte-zusammenhang`) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Bereich Graphentheorie. Vehikel: ein **Verbindungsnetz zwischen Standorten** (Depots, Umschlagpunkte, Werke), synthetisch und mit festem Zufalls-Seed, alle Leitungen gleich. Die Demo fragt: **Warum kann ein Eigenvektor einen Schnitt finden, und wie gut?** Der Eigenvektor zum zweitkleinsten Eigenwert der Laplace-Matrix (der **Fiedler-Vektor**) ordnet die Standorte; der **Sweep-Cut** schneidet an der besten Stelle der Ordnung; die **Ungleichung von Cheeger** sagt, wie gut das höchstens sein kann. Daneben stehen die vier Zielfunktionen nebeneinander: der kleinste Schnitt, **RatioCut**, **Ncut** und die **Konduktanz**.

Nachbarn: den kleinsten Schnitt ohne Balance zeigt `globaler-schnitt-demo`, die ausgewogene Halbierung durch Tausch `bisektion-demo`. Abgrenzung: `spectral-demo` nutzt den Eigenvektor als Einbettung für k-Means (Gruppenbildung, mehr als zwei Gruppen), `leiden-demo` die Modularität; hier geht es um die **Schnittsicht** auf zwei Teile mit den Maßen und Schranken, die dort nicht vorkommen.

## Warum dieses Problem

Der kleinste Schnitt eines Netzes trennt auf den Modellnetzen meist nur **einen einzelnen Standort** ab und sagt wenig über eine sinnvolle Teilung. Maße, die große Teile belohnen (RatioCut teilt durch die Zahl der Standorte, Ncut und Konduktanz durch das Volumen), wählen andere Schnitte, sind aber schwer zu minimieren. Die spektrale Antwort rechnet statt aufzuzählen: **eine** Eigenzerlegung und **n − 1** Schnitte im Sweep. Die Demo zeigt, **welches Ziel welchen Schnitt wählt**, **wie gut der Sweep das Optimum trifft**, dass die **Schranken von Cheeger ausnahmslos gelten, aber nie eng sind**, und dass **ein mehrfacher Eigenwert λ₂** (Gitter) den Fiedler-Vektor nicht eindeutig macht.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Vormessungszahlen gelten nicht:** „Min-Cut nur trivial und Ncut-Optimum nichttrivial in 100 von 122“, „Ncut- und RatioCut-Optimum fallen in 77 von 122 zusammen“, „Sweep trifft h in 88 von 122“ stammen aus Wegwerfskripten mit anderen Netzen. Die Demo misst neu (Tabelle unten): der kleinste Schnitt besteht in **59 von 84** Netzen nur aus einzelnen Standorten, der beste Ncut-Schnitt ist in **0 von 84** ein einzelner Standort; Ncut und RatioCut stimmen in **58 von 84** überein; der Sweep trifft h in **48 von 84** (normalisiert) bzw. **59 von 84** (unnormalisiert).
- **Konduktanz als viertes Ziel:** Der Plan nannte Ncut, RatioCut und Min-Cut; die Cheeger-Ungleichung gilt für die Konduktanz, deshalb steht sie als eigenes Ziel dabei (Ncut und Konduktanz haben in 50 von 84 Netzen dieselben besten Schnitte).
- **Zwei Matrizen:** Die Demo rechnet den Fiedler-Vektor wahlweise mit der **normalisierten** Matrix (passt zu Ncut und Konduktanz, Schranken von Cheeger) und der **unnormalisierten** (passt zu RatioCut); der Plan sah nur die normalisierte vor. Der Unterschied ist ein Befund (Experiment 4).
- **Entartung als neuer Befund:** Beim quadratischen Gitter ist λ₂ in **12 von 12** Netzen doppelt; der Fiedler-Vektor ist dann nicht eindeutig. Die Demo legt ihn über die Projektion eines festen Hilfsvektors auf den Eigenraum fest (unabhängig von der Basis, die die Bibliothek wählt); mit dieser Wahl trifft der normalisierte Sweep h im Gitter in **0 von 12** Netzen, der unnormalisierte in **12 von 12**. Das ist ein Befund über die Wahl im Eigenraum, kein Fehler des Verfahrens.
- **Großes Netz mit 100 statt 1.000 Standorten:** Der Vergleich im großen Netz ist hier Sweep gegen Halbierung am Median (Konduktanz), nicht gegen ein Optimum; die Zahlen der Vormessung (Gitter 58 gegen 30 usw.) sind nicht übernommen.
- **Sparsest Cut und Expander-Zerlegung nicht gebaut:** wie geplant nur Einordnung in der Formulierung.

## Modell

Ein einfaches, **zusammenhängendes** Netz mit n Standorten (Zufallsnetz: Zufallsbaum plus Leitungen bis zum mittleren Grad; Gitter, Ring mit Abkürzungen, skalenfreies Netz nach bevorzugter Anbindung, Verbundnetz mit drei Regionen und einstellbar vielen Leitungen dazwischen, Kartennetz aus zufälligen Punkten) und das handgebaute **Lehrbeispiel Hantel mit Anhängsel**: Werk 1 (A, B, C, D) und Werk 2 (E, F, G) sind je ein vollständiges Netz, dazu die Leitung D–E und der Standort H nur an A. Von Hand: der kleinste Schnitt ist **1** (H allein oder die Leitung D–E, zwei Schnitte), RatioCut, Ncut und Konduktanz wählen alle {E, F, G} gegen den Rest (Schnitt 1, Volumen 7 gegen 15: Konduktanz 1/7, Ncut 22/105, RatioCut 8/15). Das Gitter hat Quadratzahl viele Standorte (n wird gerundet). Der Zufall kommt nur aus `random.Random(seed)` mit `random()` (plattformstabil). Details und Formeln: Expander „📐 Mathematische Formulierung“ in der App.

## Methodik

- **Schnittmaße:** RatioCut = c / |S| + c / (n − |S|), Ncut = c / vol(S) + c / vol(Rest), Konduktanz φ = c / min(vol(S), vol(Rest)), mit c der Zahl der Leitungen zwischen S und dem Rest und vol der Summe der Grade.
- **Spektrum:** dichte Eigenzerlegung (`numpy.linalg.eigh`) der normalisierten Matrix I − D^(−½) A D^(−½) und der unnormalisierten D − A; λ₂ und seine Vielfachheit (Eigenwerte innerhalb 1e-9 gelten als gleich). Fiedler-Vektor: bei der normalisierten Matrix D^(−½) mal Eigenvektor, bei der unnormalisierten der Eigenvektor; in einem mehrfachen Eigenraum die Projektion eines festen Hilfsvektors.
- **Sweep-Cut:** Sortierung nach dem Fiedler-Vektor (Gleichstand, auch bei Unterschieden unter 1e-9: kleinere Nummer zuerst); jeder der n − 1 Anfänge ist ein Schnitt; gewählt wird der erste mit dem kleinsten Wert des gewählten Maßes.
- **Aufzählung:** alle 2^(n−1) − 1 Zweiteilungen (bis 18 Standorte) mit Schnitt, Größe und Volumen je Menge (vektorisiert): kleinster Wert, Zahl und Gestalt der besten Schnitte je Ziel; daraus h und die Frage, ob Ziele dieselben besten Schnitte haben.
- **Kleinster Schnitt ohne Balance:** Stoer–Wagner (siehe `globaler-schnitt-demo`), für große Netze.
- **Unabhängige Prüfer:** Die App prüft jedes Ergebnis vor der Anzeige (λ₂ positiv, Eigenwertgleichung des Fiedler-Vektors, jeder Anfang des Sweeps gleich der Nachzählung, bei höchstens 18 Standorten die Schranken von Cheeger, der Sweep zwischen h und √(2 λ₂), jeder beste Schnitt der Aufzählung mit dem gemeldeten Wert, kleinster Schnitt gleich dem der Aufzählung).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Zähler über feste Netze (je Netztyp zwölf Netze mit 16 Standorten, feste Zufalls-Seeds; im großen Netz vier je Typ mit 100 Standorten), nie ein einzelner Zufalls-Seed. Netztypen: Zufallsnetz mit mittlerem Grad 3 und 5, Gitter, Ring mit Abkürzungen, skalenfreies Netz, Verbundnetz (mittlerer Grad 5, eine Leitung zwischen je zwei Regionen), Kartennetz.

| Frage | Befund |
|---|---|
| Stimmen Spektrum, Sweep, Schranken und Aufzählung überein? | Auf **84** Netzen (sieben Netztypen, je zwölf) bestehen **1.764** Prüfungen (je Netz 21): λ₂ positiv, Eigenwertgleichung, Fiedler-Vektor senkrecht zum trivialen Eigenvektor (beide Matrizen), Schranken von Cheeger, Sweep zwischen h und √(2 λ₂), jeder Anfang des Sweeps gleich der Nachzählung, jeder Sweep nie unter dem Optimum, jeder beste Schnitt der Aufzählung mit dem gemeldeten Wert. Abweichungen: **0**. |
| Welchen Schnitt wählt welches Ziel? | Der kleinste Schnitt besteht in **59 von 84** Netzen nur aus einzelnen Standorten (in **65 von 84** ist mindestens einer der kleinsten ein einzelner Standort): Zufallsnetz Grad 3 in 10, Grad 5, Gitter, Ring und skalenfrei in je **12 von 12**, Kartennetz in 1, Verbundnetz in **0**. Der beste **Ncut**-Schnitt und der mit der kleinsten **Konduktanz** sind in **0 von 84** ein einzelner Standort, der beste **RatioCut**-Schnitt in **10 von 84** (Zufallsnetz Grad 3: 2, Grad 5: 8). |
| Wählen Ncut, RatioCut und Konduktanz dasselbe? | Ncut und RatioCut haben in **58 von 84** Netzen dieselben besten Schnitte (Gitter, Verbundnetz und Kartennetz 12 von 12, Zufallsnetz Grad 5 in **0**); Ncut und Konduktanz in **50 von 84** (Gitter und Verbundnetz 12 von 12, skalenfrei 2). |
| Gelten die Schranken von Cheeger, und wie eng sind sie? | Sie gelten in allen **84** Netzen (Verletzungen **0**). Die kleinste Konduktanz h liegt das **1,28- bis 3,08-Fache** über der unteren Schranke λ₂ / 2 und bei **15 bis 39 %** der oberen Schranke √(2 λ₂): keine der beiden ist eng. |
| Wie oft trifft der Sweep-Cut das Optimum? | Kleinste Konduktanz: normalisiert in **48 von 84**, unnormalisiert in **59 von 84** Netzen. Kleinster Ncut (normalisiert) in **58 von 84**, kleinster RatioCut (unnormalisiert) in **74 von 84**. Schlimmstenfalls liegt die Konduktanz des normalisierten Sweeps das **1,43-Fache** über h (Gitter), sonst höchstens das 1,22-Fache. |
| Was passiert bei mehrfachem λ₂? | Beim Gitter ist λ₂ in **12 von 12** Netzen doppelt; mit der Wahl der Demo trifft der normalisierte Sweep h in **0 von 12** Netzen (Ncut: 0 von 12), der unnormalisierte in **12 von 12** (RatioCut: 12 von 12). |
| Wie viele kleinste Schnitte gibt es? | Summe über je zwölf Netze: **28** (Zufallsnetz Grad 3), **19** (Grad 5), **48** (Gitter), **40** (Ring), **88** (skalenfrei), **36** (Verbundnetz), **62** (Kartennetz). |
| Sweep gegen Halbierung am Median im großen Netz? | Bei **100** Standorten (vier Netze je Typ, 28 insgesamt) liegt die Konduktanz des besten Sweep-Anfangs im Mittel bei **0,107**, die der Halbierung am Median bei **0,133**; in **18 von 28** Netzen ist der beste Anfang nicht die Hälfte. Im Verbundnetz hat die Halbierung das **10,3-Fache** der Konduktanz des Sweeps (Schnitt im Mittel 2 statt 30 Leitungen), im Kartennetz das **3,8-Fache**; im Gitter sind beide gleich. Die kleinere Seite hat im Mittel **42 %** der Standorte. |
| Das Lehrbeispiel von Hand? | Hantel mit Anhängsel, 8 Standorte, 11 Leitungen: kleinster Schnitt **1** (zwei Schnitte), Ncut, RatioCut und Konduktanz wählen {E, F, G}; Konduktanz **1/7**, Ncut **22/105**, RatioCut **8/15**; der Sweep (beide Matrizen) findet sie. |
| Sind die Preset-Beschreibungen belegt? | Ja, jedes Preset in `tests/test_claims.py` (z. B. „Der Sweep findet einen Verbund“: 90 Standorte, 30 gegen 60, 2 Leitungen, Konduktanz 0,013 gegen 0,115 für die Halbierung). |
| Ist die Rechnung korrekt? | Eigenwerte gleich `numpy.linalg.eigvalsh` auf den Matrizen von `networkx` (36 Netze, beide Matrizen), Fiedler-Vektor gleich `networkx.fiedler_vector` bis aufs Vorzeichen, Schnittmaße gleich `cut_size`, `normalized_cut_size` und `conductance` von networkx, beste Schnitte der Aufzählung gleich einer unabhängigen Aufzählung mit `itertools.combinations`, kleinster Schnitt gleich `edge_connectivity`, Eigenwerte von Kreis, Pfad und vollständigem Netz gleich den geschlossenen Formeln (`tests/test_oracle_spektral.py`). |

## Ehrliche Grenzen

- **Die Schranken sind nicht eng.** Das ist kein Mangel der Demo, sondern der Ungleichung; sie sagt nur, dass der Sweep nie schlechter als √(2 λ₂) ist, nicht dass er das Optimum trifft.
- **Der Sweep trifft das Optimum oft nicht.** Normalisiert in 48 von 84 Netzen für die Konduktanz; auf Netzen mit 16 Standorten, nicht auf großen. Welche Matrix besser ist, hängt vom Maß ab (Konduktanz, Ncut: normalisiert gedacht; RatioCut: unnormalisiert), und im Gitter zeigt sich, dass die Wahl im Eigenraum entscheidet.
- **Die Referenz im großen Netz ist die Halbierung am Median**, nicht das Optimum (keine Aufzählung über 18 Standorte); vier Netze je Typ sind eine kleine Stichprobe.
- **Dichte Eigenzerlegung bis 120 Standorte.** Für große Netze braucht man Lanczos-Verfahren oder Verfahren mit Garantie; sie sind nicht gebaut.
- **Zwei Teile, gleiche Leitungen:** Mehr als zwei Teile brauchen mehrere Eigenvektoren (`spectral-demo`), Gewichte gewichtete Summen (`globaler-schnitt-demo`). Die Teile müssen nicht zusammenhängen.
- **Aufzählung nur bis 18 Standorte** (2^17 Zweiteilungen); darüber zeigt die App Spektrum und Sweep ohne h und ohne die Zahl der besten Schnitte.
- **Grenzen der Genauigkeit:** Eigenwerte innerhalb 1e-9 gelten als gleich; bei fast mehrfachem λ₂ ohne exakte Entartung ist der Fiedler-Vektor numerisch empfindlich.

## Tests

Das Verzeichnis `tests/` enthält zurzeit 122 Tests (Laufzeit etwa 40 Sekunden auf dem Entwicklungsrechner):

- **Kern** (`test_algorithm.py`): jede Einheit an von Hand gerechneten Mini-Netzen (zwei Dreiecke, Pfad, Kreis, vollständiges Netz, Stern, Hantel): Maße, Aufzählung, Zahl der besten Schnitte, Eigenwerte und Vielfachheiten, Laplace-Matrizen, Hilfsvektor, Sweep Anfang für Anfang, Gleichstände.
- **Szenarien** (`test_scenario.py`): einfach, zusammenhängend, deterministisch für jeden Netztyp; Größen.
- **Orakel** (`test_oracle_spektral.py`): networkx (Maße, Matrizen, Fiedler-Vektor, Kantenzusammenhang), unabhängige Aufzählung, geschlossene Formeln, Ungleichung von Cheeger und Garantie des Sweeps als Eigenschaft auf 36 Netzen.
- **Randfälle** (`test_edge_cases.py`): zwei und drei Standorte, Gleichstände, Basisunabhängigkeit im entarteten Eigenraum, keine isolierten Standorte, jede Art von Prüffehler an verfälschten Analysen, Grenzen.
- **Zweige, Verdikte, Befunde** (`test_branches.py`, `test_verdicts.py`, `test_claims.py`): jedes Preset löst seine Beschreibung ein und jede Abbildung rendert; jedes Verdikt an handgesetzten Zeilen; jede Zahl dieses README.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Layout (Schnellstart oben, aktives Preset hervorgehoben), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Eigenvektor und Sweep, Welches Ziel?, Cheeger, fünf Experimente |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `spk_algorithm.py` | Kern: Schnittmaße, Aufzählung, Spektrum, Fiedler-Vektor, Sweep-Cut, Stoer–Wagner |
| `spk_constants.py` | Regler-Grenzen, Netztypen, Raster der Experimente, Farben, Presets |
| `spk_evaluation.py` | Analyse, Prüfung vor der Anzeige, Verdikte |
| `spk_experiments.py` | Experimente über feste Netze |
| `spk_layout.py` | Fester Kopf (Stylesheet) |
| `spk_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `spk_scenario.py` | Netze: Lehrbeispiel, Zufallsnetz, Gitter, Ring, skalenfreies Netz, Verbundnetz, Kartennetz |
| `spk_visualization.py` | Plotly-Abbildungen (Karten nach Fiedler-Vektor und Schnitt, sortierter Vektor, Sweep-Kurve, Cheeger-Schranken, Experimente; Achsen gesperrt) |
| `tests/` | Mini-Netze, Orakel, Randfälle, Zweige, Verdikte, Befunde, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Dünnster Schnitt mit Garantie (Arora–Rao–Vazirani), Expander-Zerlegung, Lanczos-Verfahren für große Netze:** nur als Einordnung in der Formulierung.
- **Mehr als zwei Teile, gewichtete Leitungen, gerichtete Netze:** siehe `spectral-demo` und `globaler-schnitt-demo`.
- **Echte Netze oder Dateiimport:** die Netze sind synthetisch.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
