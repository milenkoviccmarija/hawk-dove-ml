# Hawk-Dove ML

Projekat iz predmeta **Sistemi sa učenjem i samoorganizacijom (SAUSAU)** koji primenjuje mašinsko učenje na simulaciju Hawk-Dove evolutivne teorije igara.

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

### Pokretanje web UI-a

```bash
python src/app.py
```

Nakon pokretanja, aplikacija je dostupna na `http://127.0.0.1:8000`.

### Pakovanje modela za produkciju

```bash
python src/export_model.py
```

Kreira `deployment/hawk_dove_model_bundle.zip` sa modelom i metapodacima.

## Metodologija

### Pretprocesiranje

- Uklanjanje duplikata
- Podela na trening (80%) i test (20%) skup
- Skaliranje se vrši **unutar Pipeline-a** kako bi se sprečilo curenje podataka (data leakage)

### Eksplorativna analiza

- Raspodela ciljne promenljive `final_hawk`
- Korelaciona matrica numeričkih atributa
- Scatter plotovi odnosa atributa i ciljne promenljive
- Boxplotovi za detekciju outlier-a

### Modeli

Upoređena su četiri modela kroz 5-Fold Cross-Validation sa GridSearchCV:

| Model | Napomena |
|---|---|
| Dummy Regressor | Baseline — predviđa srednju vrednost |
| Ridge Regression | Linearni model sa L2 regularizacijom |
| KNN Regression | Nelinearni model baziran na susedima |
| Random Forest Regression | Ansambl stabala odlučivanja |

Svaki model je umotan u `sklearn.Pipeline` koji uključuje `StandardScaler`, čime je osigurano ispravno skaliranje tokom cross-validacije.

### Odabir najznačajnijih atributa

Korišćen je `feature_importances_` iz Random Forest modela. Izdvojena su 4 najznačajnija atributa i modeli su ponovo evaluirani, a rezultati upoređeni sa originalnim skupom atributa.

### Evaluacija

Finalni model se evaluira na izdvojenom test skupu uz sledeće metrike:
- **MAE** — srednja apsolutna greška
- **MSE** — srednja kvadratna greška  
- **RMSE** — koren srednje kvadratne greške
- **R²** — koeficijent determinacije

Generišu se i vizuelizacije: grafikon stvarnih vs. predviđenih vrednosti, grafikon reziduala i Q-Q plot.

## Rezultati

Rezultati evaluacije dostupni su u `results/metrics/evaluation_summary.txt` nakon pokretanja pipeline-a.

## Deployment

Model je dostupan kroz:
- **Web UI** (`src/app.py`) — lokalni server na portu 8000 sa formom za unos parametara
- **Python API** (`src/predict.py`) — direktno pozivanje `predict_final_hawk(scenario)` funkcije
- **ZIP bundle** (`deployment/`) — prenosivi paket sa modelom i metapodacima