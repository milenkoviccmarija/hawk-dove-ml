# Predikcija konačnog odnosa strategija u Hawk-Dove evolutivnoj igri primenom metoda mašinskog učenja

Predmet: Softverski algoritmi u sistemima automatskog upravljanja

Student: Marija Milenković, RA 89-2023

Novi Sad, jun 2026.

## Sadržaj

Opis problema

Cilj projekta

Podaci

Evaluacija i dokumentacija

## Opis problema

Evolutivne igre su matematički modeli sa različitim pravilima, dobicima i matematičkim osobinama. Svaka igra predstavlja određeni problem sa kojim se jedinke suočavaju, kao i strategije koje mogu koristiti kako bi preživele i razmnožavale se. Ove igre često imaju slikovite nazive i priče koje opisuju tipičnu situaciju. Među najpoznatijima su igra jastreb-golub, lov na jelena, tragedija zajedničkih dobara i zatvorenikova dilema. Strategije u tim igrama mogu biti jastreb, golub, buržuj, prevarant, dezerter, procenjivač ili osvetnik. Različite strategije se nadmeću prema pravilima određene igre, a matematički modeli služe za predviđanje ishoda i ponašanja sistema.

Evolutivna igra na koju ću ja primeniti mašinsko učenje je igra jastreb-golub. Ona je osmišljena kako bi objasnila sukob oko deljivog resursa. Učesnici mogu biti jastrebovi ili golubovi, odnosno dva tipa iste vrste sa različitim strategijama ponašanja.

Jastreb prvo pokazuje agresiju, a zatim eskalira sukob u pravu borbu sve dok ne pobedi ili ne bude povređen. Golub takođe pokazuje agresiju, ali ako protivnik nastavi eskalaciju, povlači se radi sigurnosti. Ako nema ozbiljne eskalacije, pokušava da podeli resurs.

Ovde je:

- V vrednost resursa,
- C trošak povrede nastale gubitkom borbe.

Pravila ishoda su sledeća:

- Ako jastreb sretne goluba, jastreb dobija ceo resurs V.
- Ako jastreb sretne drugog jastreba, svaki ima podjednake šanse za pobedu i poraz, pa je prosečan ishod V/2 - C/2.
- Ako golub sretne jastreba, povlači se i ne dobija ništa.
- Ako se sretnu dva goluba, dele resurs i svaki dobija V/2.

Stvarni dobitak zavisi od verovatnoće susreta sa jastrebovima ili golubovima, odnosno od njihovog procenta u populaciji u trenutku sukoba. Taj odnos određuju rezultati svih prethodnih sukoba.

Ako je trošak gubitka C veći od vrednosti pobede V, matematički model vodi ka evolutivno stabilnoj strategiji (ESS), odnosno stabilnoj mešavini obe strategije, gde je procenat jastrebova jednak V/C. Populacija se vraća toj ravnoteži čak i ako dođe do privremenih promena u broju jastrebova ili golubova.

Do ravnoteže se dolazi kada su fitness jastreba i fitness goluba jednaki. To znači da nijedna strategija nema prednost u odnosu na drugu, pa se njihov udeo u populaciji više ne menja. Fitness jastreba zavisi od toga koliko često sreće jastreba, a koliko često goluba. Ako je udeo jastrebova u populaciji x, onda je udeo golubova 1 - x. Kada se jastreb sretne sa jastrebom, očekivani dobitak je (V - C) / 2, a kada se sretne sa golubom, dobija V. Fitness goluba je manji u susretu sa jastrebom, jer tada dobija 0, dok u susretu sa drugim golubom dobija V/2.

U ravnoteži važi da su ova dva fitnessa jednaka. Kada se izrazi izjednače i sredi jednačina, dobija se da je udeo jastrebova u populaciji jednak odnosu vrednosti resursa i cene borbe x = V/C. To znači da će u stabilnoj populaciji procenat jastrebova biti V/C, dok će procenat golubova biti 1 - V/C. Ukoliko je jastrebova manje od V/C, njima se isplati da se šire jer imaju veći fitness. Ako ih je više od V/C, borbe postaju preskupe i njihov broj opada. Zato se sistem vraća ka odnosu V/C.

## Cilj projekta

Cilj projekta je generisanje skupa podataka simulacijom Hawk-Dove evolutivne igre i primena regresionih metoda mašinskog učenja radi predikcije konačnog udela Hawk strategije u populaciji. Cilj projekta je da ispita da li modeli mašinskog učenja mogu uspešno da nauče obrasce ponašanja populacije i aproksimiraju rezultate evolutivne simulacije na osnovu parametara koji opisuju uslove igre.

U okviru simulacije menjaće se različiti parametri, poput vrednosti resursa, cene sukoba, početnog odnosa strategija u populaciji, broja iteracija, brzine učenja, stope mutacije, promenljivosti okruženja i veličine populacije, pri čemu će se za svaku kombinaciju parametara generisati odgovarajući ishod simulacije. Na taj način biće formiran skup podataka koji će služiti za treniranje i evaluaciju regresionih modela.

Pored same predikcije konačne zastupljenosti Hawk strategije, cilj je i analiza uticaja pojedinačnih parametara evolutivne igre na ponašanje populacije i performanse modela mašinskog učenja. Projekat će sadržati kompletan proces razvoja jednog ML sistema, uključujući generisanje i pripremu podataka, izbor atributa, treniranje modela, evaluaciju rezultata i poređenje performansi različitih regresionih metoda. Na osnovu dobijenih rezultata biće analizirano u kojoj meri modeli mogu da prepoznaju stabilna stanja populacije i da generalizuju obrasce koji nastaju tokom evolutivne dinamike Hawk-Dove igre.

## Podaci

Skup podataka biće generisan izvođenjem velikog broja simulacija igre korišćenjem replikatorne dinamike sa dodatim šumom, mutacijom, promenljivošću okruženja i diskretnom veličinom populacije. Za svaku simulaciju nasumično će biti birane vrednosti parametara koji utiču na ponašanje sistema, uključujući vrednost resursa V, cenu sukoba C, početni procenat Hawk strategije u populaciji, broj iteracija simulacije, brzinu učenja, stopu mutacije, promenljivost okruženja i veličinu populacije.

Nakon pokretanja simulacije pratiće se evolucija zastupljenosti Hawk i Dove strategija kroz vreme, sve do završetka zadatog broja iteracija. Svaka simulacija predstavljaće jedan zapis u skupu podataka. Ulazne promenljive obuhvataće parametre simulacije, dok će izlazna promenljiva predstavljati konačan procenat Hawk strategije u populaciji nakon završetka simulacije. Generisanjem velikog broja simulacija sa različitim kombinacijama parametara formiraće se dovoljno veliki i raznovrstan skup podataka za treniranje i evaluaciju regresionih modela mašinskog učenja.

Primer jednog zapisa u skupu podataka može imati sledeći oblik:

| V | C | initial_hawk | iterations | learning_rate | mutation_rate | environment_volatility | population_size | final_hawk |
|---|---|--------------|------------|---------------|---------------|------------------------|-----------------|------------|
| 2 | 10 | 0.5 | 100 | 0.05 | 0.01 | 0.03 | 500 | 0.20 |

Atributi (features) koji će se koristiti kao ulazne promenljive modela su:

- vrednost resursa V,
- cena sukoba C,
- početni procenat Hawk strategije u populaciji (initial_hawk),
- broj iteracija simulacije (iterations),
- brzina učenja (learning_rate),
- stopa mutacije (mutation_rate),
- promenljivost okruženja (environment_volatility),
- veličina populacije (population_size).

Target promenljiva (label) predstavlja konačan procenat Hawk strategije u populaciji nakon završetka simulacije (final_hawk). Pošto je cilj predvideti kontinualnu numeričku vrednost, problem je formulisan kao regresioni zadatak.

## Evaluacija i dokumentacija

Pošto je problem definisan kao regresioni zadatak, evaluacija modela zasnivaće se na poređenju stvarnih vrednosti dobijenih simulacijom Hawk-Dove igre i vrednosti koje predviđa model mašinskog učenja. Za procenu performansi koristiće se standardne regresione metrike: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error) i R2 score. MAE će meriti prosečno apsolutno odstupanje predikcija od stvarnih vrednosti, MSE i RMSE će dodatno naglašavati veća odstupanja modela, dok će R2 score pokazivati koliko uspešno model objašnjava varijabilnost izlazne promenljive.

Dataset će biti podeljen na trening, validation i test skup kako bi se odvojili treniranje modela, izbor hiperparametara i finalna procena sposobnosti generalizacije modela na novim podacima.

U projektu će biti izvršeno poređenje više regresionih modela, kao što su baseline Dummy regressor, Ridge regresija, KNN regresija i Random Forest regresija, sa ciljem analize njihovih performansi nad generisanim skupom podataka. Rezultati će biti dokumentovani tabelarnim i grafičkim prikazima, kroz poređenje stvarnih i predikovanih vrednosti, prikaz evaluacionih metrika, prikaz izbora hiperparametara i analizu uticaja pojedinačnih atributa na konačno stanje populacije.

Dokumentacija projekta obuhvatiće detaljan opis generisanja skupa podataka, pretprocesinga, izbora modela, procesa treniranja i evaluacije rezultata, kao i interpretaciju dobijenih performansi modela i zaključke o uspešnosti primene metoda mašinskog učenja na problem evolutivne dinamike Hawk-Dove igre.
