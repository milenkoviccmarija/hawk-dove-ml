# Hawk-Dove ML

Projekat iz predmeta **(SAUSAU)** koji primenjuje mašinsko učenje na simulaciju Hawk-Dove evolutivne teorije igara.

## Problem

Hawk-Dove model je klasičan model evolutivne teorije igara koji opisuje kako se dve strategije — **Hawk (Jastreb)** i **Dove (Golub)** — takmiče za resurs vrednosti V uz potencijalnu cenu konflikta C. Ravnotežni udeo hawk strategije u populaciji zavisi od parametara okruženja, dinamike učenja i stohastičnih efekata.

Cilj projekta je izgradnja regresionog ML modela koji na osnovu parametara simulacije predviđa **finalni udeo hawk strategije** (`final_hawk`) u populaciji.

## Skup podataka

Podaci su generisani simulacijom (`src/main.py`) koja implementira replikator dinamiku sa sledećim parametrima:

| Atribut | Opis |
|---|---|
| `V` | Vrednost resursa |
| `C` | Cena konflikta |
| `initial_hawk` | Početni udeo hawk strategije |
| `iterations` | Broj iteracija simulacije |
| `learning_rate` | Stopa učenja |
| `mutation_rate` | Stopa mutacije |
| `environment_volatility` | Volatilnost okruženja |
| `population_size` | Veličina populacije |
| `final_hawk` | **Ciljna promenljiva** — finalni udeo hawk strategije |

Dataset sadrži **10 000 instanci** generisanih nasumičnim uzorkovanjem parametara.

## Struktura projekta

```
hawk-dove-ml/
├── data/
│   ├── raw/                    # Sirovi generisani dataset
│   └── processed/              # Train/test skupovi nakon splita
├── models/                     # Sačuvani ML pipeline
├── deployment/                 # Paket za produkciju (ZIP bundle)
├── results/
│   ├── figures/
│   │   ├── eda/                # EDA grafikoni
│   │   └── evaluation/         # Grafikoni evaluacije modela
│   └── metrics/                # Tekstualni izveštaji i metrike
└── src/
    ├── main.py                 # Generisanje dataseta simulacijom
    ├── eda.py                  # Eksplorativna analiza podataka
    ├── preprocess.py           # Čišćenje i podela podataka
    ├── train.py                # Treniranje i podešavanje hiperparametara
    ├── evaluate.py             # Finalna evaluacija na test skupu
    ├── predict.py              # Primer predikcije
    ├── export_model.py         # Pakovanje modela za deployment
    ├── app.py                  # Lokalni web UI za predikciju
    └── run_pipeline.py         # Pokretanje celog pipeline-a jednom komandom
```

## Pokretanje

### Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

### Pokretanje celog pipeline-a

```bash
python src/run_pipeline.py
```

Ovo izvršava sledeće korake redom:
1. Generisanje dataseta
2. EDA analiza
3. Pretprocesiranje i podela na train/test
4. Treniranje i podešavanje hiperparametara
5. Finalna evaluacija na test skupu
6. Primer predikcije
7. Export modela

### Pokretanje web UI-a

```bash
python src/app.py
```

Nakon pokretanja, aplikacija je dostupna na `http://127.0.0.1:8000`.
 direktno pozivanje `predict_final_hawk(scenario)` funkcije