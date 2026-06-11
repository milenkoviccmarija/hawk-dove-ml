# Hawk-Dove ML projekat

Ovaj projekat koristi simulirani Hawk-Dove model iz teorije igara i masinsko ucenje
za predikciju finalnog udela hawk strategije u populaciji.

Glavni zadatak je **regresija**, jer se predvidja kontinuirana vrednost
`final_hawk` iz intervala `[0, 1]`.

## Struktura projekta

```text
hawk-dove-ml/
├── data/
│   ├── raw/                 # generisani sirovi dataset
│   └── processed/           # skalirani train/validation/test podaci
├── models/                  # sacuvani scaler i najbolji model
├── results/
│   ├── figures/             # evaluacioni i EDA grafikoni
│   └── metrics/             # metrike i tekstualni izvestaji
├── src/
│   ├── main.py              # generisanje Hawk-Dove dataseta
│   ├── eda.py               # EDA i data wrangling izvestaji
│   ├── preprocess.py        # podela podataka i skaliranje atributa
│   ├── train.py             # treniranje, tuning i izbor modela
│   ├── evaluate.py          # evaluacija najboljeg modela
│   ├── predict.py           # primer koriscenja sacuvanog modela
│   ├── export_model.py      # eksport modela za deployment
│   └── app.py               # lokalni AI UI za unos scenarija i predikciju
└── requirements.txt
```

## Instalacija

Preporuceno je koristiti virtuelno okruzenje.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pokretanje projekta

Pokrenuti skripte ovim redosledom:

```bash
python src/main.py
python src/eda.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

## Eksport modela i AI UI

Nakon treniranja modela, deployment paket se pravi komandom:

```bash
python src/export_model.py
```

Skripta kreira folder `deployment/` sa sledecim fajlovima:

- `best_model.joblib` - eksportovani najbolji model
- `standard_scaler.joblib` - scaler potreban za iste transformacije kao u treningu
- `model_metadata.json` - opis ulaza, cilja, izlaza i metrika
- `hawk_dove_model_bundle.zip` - ZIP paket za prenos deployment fajlova

Lokalni korisnicki interfejs se pokrece komandom:

```bash
python src/app.py
```

Zatim se u browseru otvara:

```text
http://127.0.0.1:8000
```

Ako je port `8000` zauzet, aplikacija ce automatski probati sledeci slobodan
port i ispisati tacan URL u terminalu.

UI omogucava unos vrednosti atributa `V`, `C`, `initial_hawk`, `iterations`,
`learning_rate`, `mutation_rate`, `environment_volatility` i `population_size`.
Na osnovu sacuvanog modela i scalera prikazuje se predikcija `final_hawk`,
vrednost `final_dove` i dominantna strategija.

## Dataset

Dataset se generise simulacijom Hawk-Dove dinamike. Svaki red predstavlja jedan
scenario sa razlicitim parametrima okruzenja i populacije.

Najvazniji atributi:

- `V` - vrednost resursa
- `C` - cena konflikta
- `initial_hawk` - pocetni udeo hawk strategije
- `iterations` - broj iteracija simulacije
- `learning_rate` - brzina promene strategije
- `mutation_rate` - stopa mutacije strategije
- `environment_volatility` - promenljivost okruzenja
- `population_size` - velicina populacije

Ciljna promenljiva za regresiju je:

- `final_hawk` - finalni udeo hawk strategije

Klasifikaciona promenljiva koja se koristi samo za poredjenje je:

- `hawk_dominant` - vrednost 1 ako je `final_hawk > 0.5`, inace 0

## EDA i data wrangling

EDA deo se nalazi u `src/eda.py`. On pravi:

- pregled prvih redova dataseta
- proveru tipova podataka
- proveru nedostajucih vrednosti
- proveru duplikata
- deskriptivnu statistiku numerickih atributa
- raspodelu ciljne promenljive
- histograme numerickih atributa
- boxplot grafikone za proveru outlier-a
- korelacionu matricu
- grafikone odnosa atributa sa ciljnom promenljivom

Rezultati se cuvaju u:

```text
results/metrics/eda_summary.txt
results/figures/eda/
```

## Pretprocesiranje

Pretprocesiranje se nalazi u `src/preprocess.py` i obuhvata:

- uklanjanje duplikata
- izbor atributa za model
- podelu podataka na train, validation i test skup
- standardizaciju atributa pomocu `StandardScaler`
- cuvanje obradjenih skupova u `data/processed`
- cuvanje scalera u `models/standard_scaler.joblib`

Podela je:

- 70% trening skup
- 15% validation skup
- 15% test skup

## Modeli

U regresionom delu koriste se cetiri reprezentativna modela:

- Baseline Dummy Regressor
- Ridge Regression
- KNN Regression
- Random Forest Regression

Ovi modeli su izabrani zato sto pokrivaju razlicite pristupe:

- `DummyRegressor` sluzi kao baseline, odnosno najjednostavnija referenca
- `Ridge Regression` predstavlja regularizovani linearni model
- `KNN Regression` koristi slicnost izmedju scenarija evolucione igre
- `Random Forest Regression` hvata nelinearne odnose i interakcije izmedju atributa

Za Ridge se bira regularizacioni parametar `alpha`, za KNN broj suseda `k`, a za
Random Forest parametri `n_estimators`, `max_depth` i `min_samples_leaf`.

Hiperparametri se biraju na validation skupu. Svaki kandidat se fituje na
trening skupu, zatim se meri validation `RMSE` i `R2`. Za KNN se broj suseda
bira preko kolena validation krive, sto predstavlja kompromis izmedju greske i
slozenosti modela. Ridge `alpha` i Random Forest kombinacija hiperparametara
biraju se prema najmanjem validation `RMSE`.

Trenutno izabrani parametri su:

```text
Ridge alpha: 0.0001
KNN k: 5
Random Forest n_estimators: 200
Random Forest max_depth: 8
Random Forest min_samples_leaf: 3
```

## Odabir znacajnih atributa

Odabir najznacajnijih atributa radi se pomocu `feature_importances_` vrednosti iz
tuniranog Random Forest modela. Atributi se rangiraju po vaznosti, zatim se modeli
treniraju u dve varijante:

- sa svim atributima
- samo sa najznacajnijim atributima

Najznacajniji atributi su:

```text
V
C
mutation_rate
learning_rate
```

Ovakav rezultat je u skladu sa Hawk-Dove igrom: `V` i `C` direktno odredjuju
isplativost agresivne strategije, dok `mutation_rate` i `learning_rate` uticu na
dinamiku promene strategija kroz iteracije.

Rezultati poredjenja cuvaju se u:

```text
results/metrics/feature_importance.csv
results/metrics/feature_selection_results.csv
results/metrics/model_results_selected_features.csv
```

## Evaluacija

Za regresiju se koriste metrike:

- MAE
- MSE
- RMSE
- R2

Modeli se porede na validation skupu, a najbolji model se zatim ocenjuje na test
skupu. Baseline model uvek predvidja srednju vrednost ciljne promenljive iz
trening skupa i koristi se kao pocetna referentna tacka. Rezultati se cuvaju u:

```text
results/metrics/model_results.csv
results/metrics/regularization_results.csv
results/metrics/knn_neighbor_results.csv
results/metrics/random_forest_hyperparameter_results.csv
results/metrics/feature_selection_results.csv
results/metrics/results_summary.txt
results/metrics/evaluation_summary.txt
results/figures/hyperparameter_elbow.png
```

Evaluacioni grafikoni:

- stvarne vs predikovane vrednosti
- residual plot
- feature importance grafikon
- validation grafikon za izbor hiperparametara

Nalaze se u:

```text
results/figures/
```

## Koriscenje sacuvanog modela

Skripta `src/predict.py` ucitava sacuvani scaler i najbolji model, zatim pravi
predikciju za jedan novi primer ulaznih parametara. Ovaj fajl pokazuje osnovni
deployment tok bez UI-a ili API-ja.

## Trenutni rezultat

Najbolji regresioni model je:

```text
Random Forest Regression
```

Test rezultati sa svim atributima:

```text
MAE: 0.0739
MSE: 0.0095
RMSE: 0.0973
R2: 0.8834
```
