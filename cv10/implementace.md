# Cvičení 10 — Implementace pro obhajobu

Téma: **Word embeddingy, bias a debiasing** (fastText cc.cs.300, gensim)

---

## Sekce 1 – Základní operace s embeddingy

### Model a slovník

Model `cc.cs.300.vec` je fastText model trénovaný na českém Common Crawl korpusu. Načítá se přes gensim `KeyedVectors.load_word2vec_format` s limitem 200 000 slov — plný model má přes 2 miliony slov, ale pro cvičení stačí prvních 200 000 (nejfrekventovanější slova).

- **Dimenze vektoru:** 300
- **Formát:** text (.vec), ne binární
- **Velikost souboru:** ~4 310 MB

### `get_vector(word)`

Přímý přístup do `KeyedVectors` přes operátor `[]`. Vyvolá `KeyError` pokud slovo není ve slovníku — záměrně bez fallbacku, aby bylo jasné, když slovo chybí.

### `cosine_similarity(v1, v2)`

Ruční výpočet: `dot(v1, v2) / (||v1|| * ||v2||)`. Nepoužívám sklearn ani gensim vestavěnou funkci — účelem je ukázat vzorec. Ošetření dělení nulou pro nulové vektory.

Výsledky pro vybrané páry:
- `pes–kočka`: 0.598 (obě domácí zvířata)
- `pes–auto`: 0.334 (nesouvisí)
- `muž–žena`: 0.724 (oba lidé, blízký sémantický kontext)
- `radost–smutek`: 0.446 (obě emoce, ale opačná valence)

### `most_similar(vector, n, exclude_words)`

Wrapper nad `wv.similar_by_vector()` s filtrováním výsledků. Načítá `n + len(exclude_words) + 5` slov a pak filtruje, aby výsledky neobsahovaly vstupní slova.

---

## Sekce 2 – Analýza embedding prostoru

### `analogy(a, b, c, n)`

Implementuje klasický vektorový součet: `v(b) - v(a) + v(c)`. Výsledný vektor se hledá přes `most_similar` s vyloučením vstupních slov.

Výsledky analogií:
| Analogie | Výsledek | Poznámka |
|----------|----------|----------|
| Česko:Praha = Německo:? | Praha-, 1Praha... | **selhání** — šumové tokeny |
| muž:král = žena:? | královna (0.673) | výborná přesnost |
| muž:doktor = žena:? | doktorka (0.594) | funguje |
| Paříž:Francie = Berlín:? | Německa (0.539) | správně |

Geografická analogie `Česko–Praha–Německo → Berlín` selhává, protože jméno „Praha" je ve slovníku zarušené tokeny (`Praha-`, `1Praha`, `PRAHA`, `PrahaPraha`) z neočištěného webového korpusu. Tyto tokeny jsou sémanticky blíže slovu „Praha" než „Berlín".

### Vizualizace — PCA a t-SNE

40 slov rozdělených do 5 skupin: Státy, Hlavní města, Profese, Genderová slova, Příroda.

**PCA** (`sklearn.decomposition.PCA`, 2 komponenty):
- Lineární projekce, zachovává globální strukturu
- Vysvětlená variance ~10–15 % (300D → 2D, velká ztráta)
- Skupiny jsou viditelné, ale překrývají se

**t-SNE** (`sklearn.manifold.TSNE`, perplexity=15):
- Nelineární, zachovává lokální strukturu
- Skupiny jsou jasněji odděleny
- Výsledek závisí na `random_state` a `perplexity`

---

## Sekce 3 – Detekce biasu

### Bias směry

Každý bias směr je rozdíl dvou vektorů reprezentujících konce spektra:

```
b_gender = v("žena") - v("muž")         # norma 1.28
b_age    = v("starý") - v("mladý")      # norma 0.86
b_edu    = v("profesor") - v("dělník")  # norma 0.96
```

### `bias_projection(word, bias_dir)`

Projekce slova na normalizovaný bias směr:

```
b̂ = bias_dir / ||bias_dir||
projekce = v(word) · b̂
```

Kladná hodnota → blíže k kladnému pólu; záporná → k zápornému pólu.

Výsledky pro profesní slova (genderový bias, kladné = blíže k „žena"):
| Profese | Gender | Věk | Vzdělání |
|---------|--------|-----|----------|
| doktor | -0.331 | -0.156 | +0.138 |
| sestra | +0.091 | -0.038 | -0.063 |
| inženýr | -0.258 | -0.143 | -0.153 |
| kuchař | -0.303 | -0.179 | -0.239 |
| pilot | -0.363 | – | – |

Téměř všechny profese (kromě „sestra") mají záporný genderový bias → korpus asociuje profese spíše s muži.

### Stereotypní analogie

5 analogií testuje genderové stereotypy:
- `muž:doktor = žena:?` → doktorka ✓
- `muž:inženýr = žena:?` → softwarová, designérka ✗ (nevrací „inženýrka")
- `muž:vědec = žena:?` → vědkyně ✓
- `muž:šéf = žena:?` → šéfka ✓
- `žena:sestra = muž:?` → mladík, bratr ✗ (bratr až na 2. místě)

---

## Sekce 4 – Debiasing

### `debias_vector(v, bias_dir)`

Odstraní komponentu vektoru ve směru biasu — tzv. **hard debiasing** (Bolukbasi et al., 2016):

```
b̂ = bias_dir / ||bias_dir||
v' = v - (v · b̂) * b̂
```

Po operaci je projekce na bias směr přesně 0 (numericky ≈ 0.000000).

### Výsledky debiasingu jednotlivých slov

Po debiasingu slova „doktor":
- Bias projekce: -0.3306 → 0.000000
- Nejbližší sousedé se mírně mění — „doktorka" zůstává v top výsledcích, protože sémantická blízkost není jen o genderu
- „sestra" je málo ovlivněna (projekce byla jen 0.091 → blízko 0 už před)

Průměrná absolutní bias projekce skupiny profesí před debiasem: **0.2577**

---

## Sekce 5 – Zavedení biasu

### `inject_bias(v, bias_dir, lambda_val)`

Přidá bias komponentu do vektoru:

```
v' = v + λ * bias_dir
```

- `λ > 0` → posouvá ke kladnému pólu (žena)
- `λ < 0` → posouvá k zápornému pólu (muž)
- `λ = 0` → beze změny

### Experiment se slovem „doktor" (genderový bias, λ ∈ [-3, 3])

| λ | Nejbližší sousedé |
|---|-------------------|
| -3 | muž, mladík, chlapík, pán |
| -1 | muž, mladík, Doktor |
| 0 | Doktor, gynekolog, lékař, profesor |
| +1 | doktorka, gynekoložka, lékařka |
| +3 | Lektorka, doktorka, článková |

Při λ = ±3 jsou výsledky extremní — sousedé jsou generická genderová slova, nikoli profese. Vhodný rozsah pro smysluplné výsledky je λ ∈ [-1, 1].

---

## Sekce 6 – Globální manipulace prostoru

### Přístup A — globální debiasing (10 000 slov)

Debiasuje prvních 10 000 slov celého slovníku. Všechna slova mají po operaci projekci ≈ 0, ale vztahy mezi slovy jsou narušeny — debiasing ovlivňuje i slova, kde gender není relevantní.

### Přístup B — skupinový debiasing (20 profesí)

Debiasuje pouze vybranou skupinu 20 profesí. Přímý dopad jen na tato slova, zbytek slovníku zůstává nezměněn.

### Porovnání (průměrná absolutní bias projekce skupiny profesí):

| Přístup | Hodnota |
|---------|---------|
| Původní | 0.2577 |
| A — globální | 0.1113 |
| B — skupinový | 0.000000 |

Přístup B odstraní bias přesně (přímý debiasing), Přístup A jen částečně — debiasované globální vektory stále mají zbytky biasu u profesí, protože vztahy mezi slovy se vzájemně ovlivňují.

### Heatmapa cosine similarity

Porovnání 3 matic (10×10 profesí) ukazuje, že:
- Přístup A mírně zvýší průměrnou podobnost (debiasovány jsou i sousední slova)
- Přístup B zachová strukturu podobnosti, jen odstraní bias komponentu u daných slov

---

## Sekce 7 – Diskuze

### Lze odstranit bias bez ztráty informace?

**Ne.** Bias a sémantická informace jsou v embeddingách propleteny, nejsou v oddělených dimenzích. Debiasing je lineární projekce, která odstraní jeden směr ze 300 — zachová 299/300 ≈ 99,67 % variance, ale ztráta je nenulová.

Příklad: „královna" je inherentně ženské slovo. Po genderovém debiasingu ztratí část definující vlastnosti a přiblíží se slovu „král".

### Trade-off: kvalita embeddingu vs. férovost

Embeddingy jsou trénovány na maximalizaci predikce kontextu — naučí se vše ze statistiky korpusu, včetně stereotypů. Vysoce kvalitní embedding (dobrý výkon na benchmarcích) typicky zachovává i bias.

Po debiasingu:
- **Férovost se zlepší** — analogie dávají neutrálnější výsledky
- **Kvalita mírně klesne** — skóre na NLP benchmarcích (word similarity, analogy tests) se může snížit

Doporučení: debiasovat cíleně pro konkrétní citlivé atributy a konkrétní downstream kontext (např. HR systémy), ne globálně pro všechny aplikace. Hlubší, nelineární stereotypy tato technika neodstraní.

---

## Klíčové funkce — přehled

| Funkce | Účel | Poznámka |
|--------|------|----------|
| `get_vector(word)` | Získání embedding vektoru | Vyvolá KeyError pro neznámá slova |
| `cosine_similarity(v1, v2)` | Kosinová podobnost dvou vektorů | Ruční implementace vzorce |
| `most_similar(vector, n, exclude_words)` | N nejbližších sousedů | Wrapper nad gensim s filtrováním |
| `analogy(a, b, c, n)` | Vektorová analogie v(b)-v(a)+v(c) | Kontrola přítomnosti slov ve slovníku |
| `bias_projection(word, bias_dir)` | Projekce slova na bias směr | Výsledek je skalár (pozice na spektru) |
| `debias_vector(v, bias_dir)` | Hard debiasing — odstraní bias komponentu | Projekce na nulovou hodnotu |
| `inject_bias(v, bias_dir, lambda_val)` | Přidá bias do vektoru | λ řídí směr a sílu |
| `cosine_matrix(words, vector_dict)` | Matice podobností pro skupinu slov | Pro vizualizaci heatmapou |
