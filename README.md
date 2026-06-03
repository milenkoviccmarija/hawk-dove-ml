# Hawk-Dove ML projekat

Ovaj projekat koristi simulirani Hawk-Dove model iz teorije igara i masinsko ucenje
za predikciju finalnog udela hawk strategije u populaciji.

Glavni zadatak je **regresija**, jer se predvidja kontinuirana vrednost
`final_hawk` iz intervala `[0, 1]`. U projektu postoji i jedan manji
klasifikacioni primer za cilj `hawk_dominant`, da bi se pokazalo zasto je
regresija informativnija za ovaj problem.

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
│   ├── train.py             # treniranje, CV i izbor modela
│   ├── evaluate.py          # evaluacija najboljeg modela
│   └── predict.py           # primer koriscenja sacuvanog modela
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

U regresionom delu koriste se modeli koji su radjeni na vezbama:

- Baseline Dummy Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regression

Za Ridge i Lasso se bira regularizacioni parametar `alpha`, odnosno lambda.
Za KNN regresiju se bira broj suseda `k`.

Parametri se biraju pomocu 5-fold unakrsne validacije na trening skupu. Za svaku
kandidatsku vrednost racuna se prosecan `R2` skor, a bira se vrednost sa najboljim
prosecnim rezultatom.

Trenutno izabrani parametri su:

```text
Ridge alpha: 0.1
Lasso alpha: 0.001
KNN k: 15
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
results/metrics/results_summary.txt
results/metrics/evaluation_summary.txt
```

Evaluacioni grafikoni:

- stvarne vs predikovane vrednosti
- residual plot

Nalaze se u:

```text
results/figures/
```

## Koriscenje sacuvanog modela

Skripta `src/predict.py` ucitava sacuvani scaler i najbolji model, zatim pravi
predikciju za jedan novi primer ulaznih parametara. Ovaj fajl pokazuje osnovni
deployment tok bez UI-a ili API-ja.

## Klasifikacioni primer

U `src/train.py` dodat je jedan primer klasifikacije pomocu logisticke regresije.
Cilj je `hawk_dominant`, odnosno da li je finalni udeo hawk strategije veci od
0.5.

Ovaj deo pokazuje razliku izmedju regresije i klasifikacije:

- klasifikacija daje samo klasu: hawk ili dove
- regresija daje tacan procenjeni udeo `final_hawk`

Zbog toga je regresija glavni zadatak ovog projekta.

## Trenutni rezultat

Najbolji regresioni model je:

```text
KNN Regression (k=15)
```

Test rezultati:

```text
MAE: 0.0937
MSE: 0.0149
RMSE: 0.1222
R2: 0.8159
```

Klasifikacioni primer sa logistickom regresijom ima test accuracy oko `0.9173`,
ali on resava jednostavniji zadatak jer predvidja samo dominantnu strategiju.
