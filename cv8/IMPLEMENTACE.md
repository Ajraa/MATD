# CV8 — Implementace překladu slov pomocí lineárního mapování embeddingů

## Obsah

1. [Úvod a cíl](#1-úvod-a-cíl)
2. [Teoretický základ](#2-teoretický-základ)
3. [Část 1 — Příprava dat](#3-část-1--příprava-dat)
4. [Část 2 — Trénování](#4-část-2--trénování)
5. [Část 3 — Překlad a vyhodnocení](#5-část-3--překlad-a-vyhodnocení)
6. [Tok dat — end-to-end přehled](#6-tok-dat--end-to-end-přehled)
7. [Otázky ze zadání a jejich odpovědi](#7-otázky-ze-zadání-a-jejich-odpovědi)
8. [Klíčové implementační detaily a možná úskalí](#8-klíčové-implementační-detaily-a-možná-úskalí)

---

## 1. Úvod a cíl

### Problém

Přeložit slovo ze zdrojového jazyka (čeština) do cílového jazyka (angličtina) bez tradičního slovníku je netriviální úkol. Jeden z přístupů využívá skutečnosti, že jazykové prostory embeddingů mají podobnou geometrickou strukturu — sémantické vztahy mezi slovy jsou v různých jazycích analogické. Empiricky platí: pokud v češtině platí $\vec{\text{pes}} - \vec{\text{kočka}} \approx \vec{\text{vlk}} - \vec{\text{liška}}$, podobná relace platí i v angličtině.

Z tohoto pozorování plyne idea: existuje-li lineární zobrazení mezi českým a anglickým embedding prostorem, stačí ho nalézt a pak libovolný český vektor transformovat do angličtiny — a poté vyhledat nejbližší anglické slovo.

### Metoda MUSE (supervised varianta)

Toto cvičení implementuje **supervised** variantu metody MUSE (Multilingual Unsupervised and Supervised Embeddings). Vstupy jsou:

- Předtrénované fastText embeddingy pro češtinu a angličtinu (soubory `.vec`)
- Dvojjazyčný trénovací slovník (soubor `.txt` s dvojicemi `cs_slovo → en_slovo`)

Trénink najde matici $W^T$ takovou, aby $X W^T \approx Y$, kde $X$ jsou české a $Y$ anglické embeddingy slov ze slovníku.

### Co je výsledkem

Naučená transformační matice $W^T \in \mathbb{R}^{300 \times 300}$ — lineární zobrazení z českého embedding prostoru do anglického. Po natrénování na 6 567 párech a vyhodnocení na 1 770 testovacích párech dosahuje implementace:

- **Accuracy@1 = 42,09 %** — správný překlad je první výsledek
- **Accuracy@5 = 67,80 %** — správný překlad je mezi prvními 5 výsledky

Výsledky jsou srovnatelné s literaturou pro supervised metody s tímto rozsahem dat.

---

## 2. Teoretický základ

### 2.1 Co je embedding

**Embedding** (vektorová reprezentace) je mapování diskrétního objektu — typicky slova — do spojitého vektorového prostoru $\mathbb{R}^d$. Každé slovo je reprezentováno hustým vektorem reálných čísel o $d$ dimenzích (v tomto cvičení $d = 300$).

Embeddingy jsou trénovány tak, aby slova, která se vyskytují v podobných kontextech, měla blízké vektory. Tato idea je formalizována jako **distributional hypothesis**: *slova, která se vyskytují ve stejných kontextech, mají podobný význam* (Firth, 1957).

Praktickým důsledkem je, že embeddingy zachycují sémantické a syntaktické vztahy. Klasický příklad:

$$\vec{\text{král}} - \vec{\text{muž}} + \vec{\text{žena}} \approx \vec{\text{královna}}$$

### 2.2 Word2Vec

Word2Vec (Mikolov et al., 2013) trénuje embeddingy pomocí jednoduché neuronové sítě na dvou úlohách:

- **CBOW** (Continuous Bag of Words): z kontextu předpovídá centrální slovo
- **Skip-gram**: z centrálního slova předpovídá kontext

Nevýhoda: slova, která nebyla v trénovacím korpusu (**OOV slova** — Out Of Vocabulary), nemají žádný vektor.

### 2.3 FastText a subword n-gramy

FastText (Bojanowski et al., 2017) rozšiřuje Word2Vec tím, že každé slovo rozkládá na **subword n-gramy** (části slov). Vektor slova je pak součtem vektorů jeho n-gramů. Například slovo `"Praha"` může být rozloženo na `"<Pr"`, `"Pra"`, `"rah"`, `"aha"`, `"ha>"` apod.

Výhoda: fastText dokáže vygenerovat smysluplný vektor i pro slovo, které nikdy neviděl — stačí, aby znal jeho části. Tím řeší OOV problém.

**Avšak** — ve formátu `.vec`, který toto cvičení používá, jsou uloženy pouze předpočítané vektory slov ze slovníku. Subword model **není dostupný**, takže OOV slova stále vektory nemají. Cvičení proto musí páry s OOV slovy přeskakovat.

### 2.4 Formát `.vec`

Soubory `.vec` jsou textový formát kompatibilní s word2vec. Struktura:

```
200000 300
,  0.0052  0.1646  0.0675 ... -0.0156
.  0.0485  0.0674  0.0261 ...  0.0116
```

- **První řádek**: `vocab_size dim` — velikost slovníku a dimenze vektoru
- **Každý další řádek**: `slovo f_1 f_2 ... f_d` — slovo následované $d$ čísly plovoucí desetinné čárky

Vektory jsou uloženy jako `float32` — dostatečná přesnost pro tuto aplikaci při polovičních paměťových nárocích oproti `float64`.

### 2.5 Kosinová podobnost

Pro porovnávání vektorů se v NLP standardně používá **kosinová podobnost** namísto Euklidovské vzdálenosti. Definice:

$$\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

Výsledek leží v intervalu $[-1, 1]$:
- $1$ — vektory míří stejným směrem (maximální podobnost)
- $0$ — vektory jsou kolmé (žádná podobnost)
- $-1$ — vektory míří proti sobě

**Proč ne Euklidovská vzdálenost?** Délka vektoru embeddingu závisí na frekvenci slova v korpusu — frekventovaná slova mají tendenci k delším vektorům. Kosinová podobnost délku normalizuje a srovnává pouze **směr** vektoru, který nese sémantický obsah. Díky tomu "auto" bude blíže "car" než "vehicle" správně i přesto, že "car" je frekventovanější a má jinak dlouhý vektor.

---

## 3. Část 1 — Příprava dat

### 3.1 `load_vec_model` — načtení fastText vektorů

```python
def load_vec_model(lang_code, max_words=200000):
    model_path = f'cc.{lang_code}.300.vec'
    words = []
    vectors = []
    with open(model_path, 'r', encoding='utf-8') as f:
        vocab_size, dim = map(int, f.readline().strip().split())
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(' ')
            words.append(parts[0])
            vectors.append(np.array(parts[1:], dtype=np.float32))
    matrix = np.array(vectors, dtype=np.float32)
    word2vec = {w: matrix[i] for i, w in enumerate(words)}
    return word2vec, words, matrix
```

**Průběh parsování:**

1. Otevře soubor s kódováním UTF-8 (nutné pro diakritiku)
2. Přečte první řádek a z něj extrahuje `vocab_size` a `dim`
3. Iteruje po řádcích, každý řádek rozdělí na slovo a seznam čísel
4. Čísla převede na `np.array` typu `float32`
5. Omezení `max_words=200000` — úplný fastText slovník má stovky tisíc až miliony slov; nahrávat vše do paměti by bylo zbytečně nákladné a pro toto cvičení zbytečné

**Návratové hodnoty:**

| Hodnota | Typ | Popis |
|---|---|---|
| `word2vec` | `dict {str → np.array(300)}` | Rychlý lookup: slovo → vektor |
| `words` | `list[str]` | Slova v pořadí ze souboru (pro mapování indexu → slovo) |
| `matrix` | `np.array (200000, 300)` | Všechny vektory v matici (pro maticové operace při překladu) |

**Proč vracíme jak dict, tak list, tak matici?**
- `word2vec` — O(1) lookup při sestavování trénovacích matic a při překladu vstupního slova
- `words` — po nalezení nejbližšího indexu v matici potřebujeme zpět slovo
- `matrix` — efektivní maticová operace: kosinová podobnost jednoho vektoru vůči všem 200 000 anglickým slovům najednou pomocí `tgt_matrix @ translated`

### 3.2 `load_dictionary` — načtení překladového slovníku MUSE

```python
def load_dictionary(filepath):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(' ', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs
```

**MUSE formát** jsou jednoduché textové soubory, kde každý řádek obsahuje jeden překlad. Oddělovač je buď tabulátor nebo mezera — kód zvládá oboje. Argument `maxsplit=1` zajistí, že i víceslovné překlady (`"New York"`) se zpracují správně jako jeden cílový výraz.

**Vstupní soubory:**
- `cs-en.0-5000.txt` — 7 349 trénovacích dvojic
- `cs-en.5000-6500.txt` — 2 071 testovacích dvojic

Čísla v názvech souborů jsou offsety v původním MUSE slovníku — train a test sady jsou tedy disjunktní podmnožiny jednoho velkého slovníku.

### 3.3 `build_matrices` — sestavení trénovacích matic

```python
def build_matrices(src_word2vec, tgt_word2vec, pairs):
    X_rows = []
    Y_rows = []
    skipped = 0
    for src_word, tgt_word in pairs:
        if src_word not in src_word2vec or tgt_word not in tgt_word2vec:
            skipped += 1
            continue
        X_rows.append(src_word2vec[src_word])
        Y_rows.append(tgt_word2vec[tgt_word])
    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    return X, Y
```

**OOV filtrování:** Každá dvojice prochází testem přítomnosti v obou `word2vec` slovnících. Pokud buď české nebo anglické slovo chybí (bylo mimo prvních 200 000 slov, nebo šlo o velmi vzácný výraz), dvojice se přeskočí. Z trénovacích 7 349 párů se přeskočí 782 (≈ 10,6 %) a z testovacích 2 071 párů se přeskočí 327 (≈ 15,8 %).

**Proč se OOV páry přeskakují, ne doplňují?** Soubor `.vec` neobsahuje subword model — nelze vytvořit vektor pro neznámé slovo na základě jeho částí. Jedinou alternativou by bylo použít celý binární fastText model (`.bin`), který subword model obsahuje, ale to výrazně zvyšuje paměťové nároky.

**Výsledné tvary matic:**

| Matice | Tvar | Obsah |
|---|---|---|
| `X_train` | `(6567, 300)` | České embeddingy trénovacích slov |
| `Y_train` | `(6567, 300)` | Anglické embeddingy odpovídajících překladů |
| `X_test` | `(1744, 300)` | České embeddingy testovacích slov |
| `Y_test` | `(1744, 300)` | Anglické embeddingy testovacích překladů |

Klíčová vlastnost: $X[i]$ a $Y[i]$ jsou vždy vektory překladového páru — tato korespondence je základem tréninku.

---

## 4. Část 2 — Trénování

### 4.1 Model

Hledáme matici $W^T \in \mathbb{R}^{300 \times 300}$ takovou, aby platilo:

$$X W^T \approx Y$$

kde:
- $X \in \mathbb{R}^{n \times 300}$ — embeddingy zdrojových slov (čeština), $n$ = počet trénovacích párů
- $Y \in \mathbb{R}^{n \times 300}$ — embeddingy cílových slov (angličtina)
- $W^T \in \mathbb{R}^{300 \times 300}$ — hledaná transformační matice

Zápis $W^T$ (s transpozicí) je notační konvence z původní MUSE literatury: matice $W \in \mathbb{R}^{300 \times 300}$ je ortogonální projekce, a model je $X W^T \approx Y$.

Operace $X W^T$ má tvar $(n \times 300) \cdot (300 \times 300) = (n \times 300)$ — pro každý z $n$ párů transformuje 300-dimenzionální český vektor na 300-dimenzionální anglický vektor.

### 4.2 Loss funkce — Frobeniova norma

Minimalizujeme **čtvercovou Frobeniovu normu** residuální matice:

$$L(W^T) = \|X W^T - Y\|_F^2 = \sum_{i=1}^{n} \sum_{j=1}^{300} \left(X W^T - Y\right)_{ij}^2$$

**Proč Frobeniova norma?** Jde o nejpřirozenější rozšíření metody nejmenších čtverců na matice. Každý prvek residuální matice $(XW^T - Y)_{ij}$ odpovídá chybě na $j$-té dimenzi $i$-tého překladového páru. Suma čtverců všech těchto chyb dává jednu skalární hodnotu, kterou chceme minimalizovat.

Ekvivalentní vyjádření přes stopu:

$$L = \|D\|_F^2 = \mathrm{trace}(D^T D)$$

kde $D = X W^T - Y$ je residuální matice.

### 4.3 Odvození gradientu

Chceme $\nabla_{W^T} L$ — matici parciálních derivací $L$ podle každého prvku $W^T$.

**Krok 1: Označení**

$$D = X W^T - Y \quad \in \mathbb{R}^{n \times 300}$$

$$L = \|D\|_F^2 = \sum_{i,j} D_{ij}^2$$

**Krok 2: Diferenciál**

Spočítáme diferenciál $dL$ při perturbaci $dW^T$:

$$dD = X \, dW^T$$

$$dL = \mathrm{trace}\!\left(D^T \, dD + dD^T D\right) = 2 \, \mathrm{trace}\!\left(D^T X \, dW^T\right)$$

Protože $L$ je skalár, platí $dL = \mathrm{trace}\!\left((\nabla_{W^T} L)^T dW^T\right)$, takže:

$$\nabla_{W^T} L = \left(D^T X\right)^T = X^T D$$

**Výsledek:**

$$\boxed{\nabla_{W^T} L = 2 X^T (X W^T - Y)}$$

**Kontrola rozměrů:**

$$X^T \in \mathbb{R}^{300 \times n}, \quad D \in \mathbb{R}^{n \times 300} \implies X^T D \in \mathbb{R}^{300 \times 300} \checkmark$$

Gradient má stejný tvar jako $W^T$ — to je nutná podmínka pro update pravidlo.

Stručnější odvození přes maticové pravidlo: pro $L = \|AZ - B\|_F^2$ platí $\nabla_Z L = 2A^T(AZ - B)$. V našem případě $A = X$, $Z = W^T$, $B = Y$.

### 4.4 Gradient descent update

$$W^T \leftarrow W^T - \alpha \cdot \nabla_{W^T} L = W^T - \alpha \cdot 2 X^T (X W^T - Y)$$

kde $\alpha$ je **learning rate** (krok učení). Implementace:

```python
def compute_gradient(X, W_T, Y):
    D = X @ W_T - Y          # residuál, tvar (n, 300)
    return 2 * X.T @ D       # gradient, tvar (300, 300)

def gradient_descent_step(W_T, gradient, alpha):
    return W_T - alpha * gradient
```

### 4.5 Implementace `train` — detailní průběh

```python
def train(X_train, Y_train, alpha=0.01, max_steps=1000,
          convergence_window=10, min_improvement=1e-6):
    n_src = X_train.shape[1]   # 300
    n_tgt = Y_train.shape[1]   # 300

    # Inicializace
    W_T = np.random.randn(n_src, n_tgt).astype(np.float32) * 0.01

    loss_history = []

    for step in range(max_steps):
        loss = frobenius_norm_squared(X_train, W_T, Y_train)
        loss_history.append(loss)

        # Detekce konvergence
        if len(loss_history) > convergence_window:
            recent_improvement = loss_history[-convergence_window - 1] - loss_history[-1]
            if recent_improvement < min_improvement:
                print(f"Konvergence dosažena v kroku {step}")
                break

        grad = compute_gradient(X_train, W_T, Y_train)
        W_T = gradient_descent_step(W_T, grad, alpha)

    return W_T, loss_history
```

**Inicializace malými náhodnými hodnotami (`* 0.01`):**

Inicializace nulami by vedla k problému: gradient $2 X^T (X \cdot 0 - Y) = -2 X^T Y$ by byl sice nenulový, ale všechny prvky $W^T$ by se pohybovaly synchronně ve stejném směru. To technicky funguje pro lineární modely, ale malá náhodná inicializace je robustnější konvencí — zabrání případným symetriím a lépe odpovídá praktikám hlubokého učení, kde by nulová inicializace způsobila kolaps.

**Detekce konvergence — sliding window:**

Nestačí porovnat loss dvou sousedních kroků, protože loss může v krátkodobém horizontu oscilovat. Proto se porovnává loss nyní ($t$) s hodnotou před `convergence_window` kroky ($t - k$). Podmínka:

$$L_{t-k} - L_t < \text{min\_improvement}$$

Pokud za posledních 10 kroků klesla loss o méně než $10^{-6}$, trénink skončí. Tím se předchází zbytečným iteracím po dosažení plató.

**Parametry při spuštění:**

```python
W_T, loss_history = train(
    X_train, Y_train,
    alpha=1e-4,       # learning rate
    max_steps=500,    # maximální počet kroků
    convergence_window=10,
    min_improvement=1e-6
)
```

**Průběh loss:**

| Krok | Loss |
|---|---|
| 0 | 7 717,05 |
| 100 | 5 364,68 |
| 200 | 4 978,37 |
| 300 | 4 809,14 |
| 400 | 4 719,04 |

Loss klesá, ale pomalu — to je typické pro gradient descent na velkých maticích s malým `alpha`. Po 500 krocích loss stále klesá (nedosáhla konvergence), ale výsledky jsou již použitelné.

**Proč je `alpha = 1e-4` citlivý parametr?**

Gradient je $2 X^T D$. Matice $X$ má tvar $(6567, 300)$ a obsahuje hodnoty řádu $10^{-1}$ (fastText vektory jsou normalizované). Gradient má proto normu řádu $\sim n \cdot d \cdot 10^{-2} \approx 10^4$. Aby byl update $W^T$ stabilní (pohyboval se o řád $10^{-2}$ až $10^{-1}$ per krok), potřebujeme $\alpha \approx 10^{-4}$ až $10^{-6}$.

Příliš velké `alpha` (např. $10^{-2}$) způsobí divergenci — loss místo klesání roste. Příliš malé `alpha` (např. $10^{-7}$) zase znamená, že by bylo potřeba řádově více kroků pro dosažení stejné kvality.

---

## 5. Část 3 — Překlad a vyhodnocení

### 5.1 `translate_word` — překlad jednoho slova

```python
def translate_word(word, src_word2vec, W_T, tgt_matrix, tgt_words, top_k=5):
    if word not in src_word2vec:
        return []

    src_vec = src_word2vec[word].astype(np.float32)   # (300,)
    translated = src_vec @ W_T                          # (300,)

    norms = np.linalg.norm(tgt_matrix, axis=1)         # (200000,)
    translated_norm = np.linalg.norm(translated)        # skalár

    if translated_norm == 0:
        return []

    similarities = tgt_matrix @ translated / (norms * translated_norm + 1e-10)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(tgt_words[i], float(similarities[i])) for i in top_indices]
```

**Krok za krokem:**

**1. OOV check:** Pokud vstupní slovo není ve `src_word2vec`, vrátí prázdný seznam. Nelze překládat slova, pro která nemáme vektor.

**2. Lookup zdrojového vektoru:**

```python
src_vec = src_word2vec[word].astype(np.float32)  # tvar (300,)
```

Explicitní přetypování na `float32` zajistí konzistenci s `W_T`, která je také `float32`.

**3. Transformace do cílového prostoru:**

```python
translated = src_vec @ W_T  # (300,) @ (300, 300) = (300,)
```

Výsledek `translated` je 300-dimenzionální vektor v **anglickém** embedding prostoru — zatím nezávislý na slovníku.

**4. Maticový výpočet kosinové podobnosti:**

```python
norms = np.linalg.norm(tgt_matrix, axis=1)         # norma každého ang. slova, tvar (200000,)
similarities = tgt_matrix @ translated / (norms * translated_norm + 1e-10)
```

Rozepsáno po prvcích: $\text{similarities}[i] = \frac{\mathbf{m}_i \cdot \mathbf{t}}{\|\mathbf{m}_i\| \cdot \|\mathbf{t}\|}$, kde $\mathbf{m}_i$ je $i$-tý řádek `tgt_matrix` a $\mathbf{t}$ je `translated`.

Klíčová optimalizace: `tgt_matrix @ translated` je jedna maticová operace $(200000, 300) \cdot (300,) = (200000,)$ — výpočetně řádově rychlejší než smyčka přes 200 000 slov.

**5. Výběr top-k výsledků:**

```python
top_indices = np.argsort(similarities)[-top_k:][::-1]
```

`np.argsort` vrátí indexy seřazené od nejmenší do největší podobnosti. `[-top_k:]` vezme posledních 5 (nejvyšší podobnosti), `[::-1]` je obrátí od nejvyšší po nejnižší.

**Ukázka výsledků:**

| CZ slovo | Top 5 anglických překladů |
|---|---|
| pes | dog (0.831), cat (0.702), dogs (0.695), pup (0.686), puppy (0.665) |
| auto | car (0.886), cars (0.739), vehicle (0.717), truck (0.706), automobile (0.668) |
| Praha | europe (0.533), city. (0.526), london (0.515), berlin (0.508), amsterdam (0.508) |

Pozorování: Obecná slova jako "pes" nebo "auto" se překládají výborně (nejvyšší podobnost > 0.8). Vlastní jméno "Praha" je problematičtější — model zobrazuje do sémantické sousednosti (evropská hlavní města), ale nenajde přesný ekvivalent "Prague".

### 5.2 `evaluate_accuracy` — vyhodnocení přesnosti

```python
def evaluate_accuracy(test_pairs, src_word2vec, W_T, tgt_matrix, tgt_words, top_k=5):
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for src_word, expected_tgt in test_pairs:
        results = translate_word(src_word, src_word2vec, W_T, tgt_matrix, tgt_words, top_k=5)
        if not results:
            continue
        predicted_words = [w for w, _ in results]

        total += 1
        if predicted_words[0] == expected_tgt:
            correct_top1 += 1
        if expected_tgt in predicted_words:
            correct_top5 += 1

    acc1 = correct_top1 / total if total > 0 else 0.0
    acc5 = correct_top5 / total if total > 0 else 0.0
    return acc1, acc5, total
```

**Accuracy@1 vs Accuracy@5:**

- **Accuracy@1** — nejpřísnější metrika: správný překlad musí být úplně první výsledek. Měří, zda model přeloží přesně.
- **Accuracy@5** — benevolentnější: správný překlad musí být kdekoli v prvních 5 výsledcích. Realisticky odpovídá scénáři, kde uživatel dostane návrhy a vybere správný.

Rozdíl 42,09 % vs 67,80 % (tj. +25,71 %) ukazuje, že model správný překlad "ví", ale nemusí ho vždy zařadit na první místo — v sousedství správného překladu jsou blízká synonyma nebo morfologické varianty (dog/dogs, car/cars).

**Proč se přeskakují OOV páry (`if not results: continue`)?**

OOV slova nejsou spravedlivou testovací zátěží — model pro ně nemá vstup, takže jejich zahrnutí do jmenovatele by uměle snižovalo accuracy. Místo toho se `total` inkrementuje pouze pro slova s dostupným vektorem. Z 2 071 testovacích párů je `total = 1770` (zbylých 301 je OOV nebo přeskočeno jinak).

**Výsledky:**

```
Testovacích dvojic vyhodnoceno : 1770
Accuracy@1                     : 0.4209  (42.09 %)
Accuracy@5                     : 0.6780  (67.80 %)
```

---

## 6. Tok dat — end-to-end přehled

```
Soubory na disku
════════════════
cc.cs.300.vec          cc.en.300.vec
(200k × 300 float32)   (200k × 300 float32)
        │                      │
        ▼                      ▼
  load_vec_model('cs')   load_vec_model('en')
        │                      │
        ├──────────────────────┤
        │    cs_word2vec       │    en_word2vec
        │    cs_words          │    en_words
        │    cs_matrix         │    en_matrix
        │                      │
        └──────────┬───────────┘
                   │
        cs-en.0-5000.txt          cs-en.5000-6500.txt
              │                           │
              ▼                           ▼
        load_dictionary()         load_dictionary()
              │                           │
        train_pairs (7349)        test_pairs (2071)
              │                           │
              ▼                           ▼
        build_matrices()          build_matrices()
              │                           │
        X_train (6567×300)        X_test (1744×300)
        Y_train (6567×300)        Y_test (1744×300)
              │
              ▼
           train()
        ┌─────────────────────────────────────┐
        │  W_T = randn(300,300) * 0.01        │
        │  for step in range(500):            │
        │    loss = ||X @ W_T - Y||_F²        │
        │    if converged: break              │
        │    grad = 2 * X.T @ (X @ W_T - Y)  │
        │    W_T -= alpha * grad             │
        └─────────────────────────────────────┘
              │
        W_T (300×300)   ←── naučená transformační matice
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
translate_word()  evaluate_accuracy()
     │                 │
  "pes"             Acc@1: 42.09 %
     │               Acc@5: 67.80 %
     ▼
  src_vec = cs_word2vec["pes"]       # (300,)
  translated = src_vec @ W_T         # (300,)
  sims = en_matrix @ translated /    # (200000,)
         (norms * |translated|)
  top5 = argsort(sims)[-5:][::-1]
     │
  [("dog", 0.831), ("cat", 0.702), ...]
```

---

## 7. Otázky ze zadání a jejich odpovědi

### 7.1 Co je embedding?

**Embedding** je zobrazení diskrétního objektu (slova) do spojitého vektorového prostoru $\mathbb{R}^d$. V kontextu tohoto cvičení:

- Každé slovo je reprezentováno hustým vektorem 300 reálných čísel
- Vektory jsou natrénované tak, aby sémanticky příbuzná slova měla geometricky blízké vektory (distributional hypothesis)
- FastText embeddingy jsou rozšířením word2vec — trénují se na subword n-gramech, takže zvládají OOV slova (v binárním formátu; `.vec` soubory tuto schopnost nemají)
- Praktický důsledek: embedding zachycuje analogie jako $\vec{\text{král}} - \vec{\text{muž}} + \vec{\text{žena}} \approx \vec{\text{královna}}$

### 7.2 Co dělá gradient descent?

**Gradient descent** je iterativní algoritmus pro minimalizaci účelové funkce:

1. Inicializuj parametry (zde: $W^T$ malými náhodnými hodnotami)
2. Vypočítej gradient $\nabla_{W^T} L$ — ukazuje směr nejstrmějšího růstu loss
3. Aktualizuj parametry v opačném směru: $W^T \leftarrow W^T - \alpha \nabla_{W^T} L$
4. Opakuj až do konvergence

**Volba learning rate $\alpha$:**
- Příliš velké $\alpha$ → oscilace, divergence (loss roste)
- Příliš malé $\alpha$ → pomalá konvergence, zbytečně mnoho iterací
- Pro fastText vektory dimenze 300 je doporučená hodnota $\alpha \approx 10^{-4}$

**Intuice:** Gradient descent je jako hledání nejnižšího bodu v krajině naslepo — v každém kroku sáhneme pod nohy, zjistíme sklon terénu a uděláme krok dolů po svahu.

### 7.3 Jak by se změnily X a Y při použití W místo W^T?

Aktuální model:

$$X W^T \approx Y, \quad W^T \in \mathbb{R}^{300 \times 300}$$

Pokud bychom přejmenovali $W^T \to W$ (tedy $W \in \mathbb{R}^{300 \times 300}$), model by zůstal:

$$X W \approx Y$$

Matice $X$ a $Y$ by zůstaly **beze změny** — jde pouze o notační přejmenování parametru.

Pokud bychom chtěli alternativní formulaci, kde figuruje matice $W \in \mathbb{R}^{300 \times 300}$ a kde $W$ je fyzicky transponovaná oproti výše (tedy $W^T$ v novém smyslu = původní $W$), mohli bychom psát:

$$W X^T \approx Y^T$$

V tomto případě by $X$ a $Y$ musely být **transponovány**: $X^T \in \mathbb{R}^{300 \times n}$, $Y^T \in \mathbb{R}^{300 \times n}$. Výsledek je matematicky ekvivalentní — oba zápisy hledají stejné lineární zobrazení, jen z jiné perspektivy.

**Závěr:** Samotné použití $W$ místo $W^T$ (jako název parametru) nevyžaduje žádnou změnu $X$ ani $Y$. Změna matic by nastala pouze při přechodu na transponovanou formulaci $WX^T \approx Y^T$.

---

## 8. Klíčové implementační detaily a možná úskalí

### 8.1 Proč `float32`

Celý pipeline používá `dtype=np.float32` konzistentně:

- **Paměťová úspora:** `float32` zabírá 4 bajty oproti 8 bajtům u `float64`. Při matici 200 000 × 300 to znamená rozdíl 240 MB vs 480 MB pro jeden jazykový model. Dohromady (CZ + EN) `float32` šetří přibližně 480 MB RAM.
- **Kompatibilita s fastText:** Původní fastText model ukládá vektory v `float32`. Přetypování na `float64` by bylo ztrátová konverze (zdánlivě přesnější, ale informaci nepřidá).
- **Výkon:** NumPy maticové operace na `float32` jsou na moderním hardware rychlejší díky SIMD instrukcím a menší zátěži cache.

### 8.2 Proč `+ 1e-10` při dělení v kosinové podobnosti

```python
similarities = tgt_matrix @ translated / (norms * translated_norm + 1e-10)
```

Přidání malé konstanty $10^{-10}$ brání dělení nulou v případě, že norma `translated` nebo norma některého cílového vektoru je nula. V praxi se nulový vektor může objevit v okrajových případech (poškozený vstup, numerická podtečení). Explicitní kontrola `if translated_norm == 0: return []` sice zachytí nulový výsledek transformace, ale `norms` (normy 200 000 anglických slov) se nekontrolují individuálně — proto `+ 1e-10` v jmenovateli pokrývá i tento případ bez výrazného vlivu na výsledek (normy jsou typicky řádu $10^0$ až $10^1$).

### 8.3 OOV problém a jak ho kód řeší

Formát `.vec` obsahuje pouze statické předpočítané vektory. Bez binárního fastText modelu (`.bin`) nelze generovat vektory pro neznámá slova.

Kód řeší OOV na třech místech:

| Místo | Řešení |
|---|---|
| `build_matrices` | Přeskočí dvojice, kde chybí CS nebo EN slovo — není zahrnuto do tréninku ani testování |
| `translate_word` | Vrátí `[]` pro OOV vstup |
| `evaluate_accuracy` | OOV vstup způsobí `if not results: continue` — přeskočí se, ale nezapočítá se jako chyba |

Tímto přístupem se vyhýbáme "falešným chybám" způsobeným absencí dat, nikoli špatnou kvalitou modelu.

### 8.4 Citlivost na learning rate

Praktická doporučení pro volbu `alpha`:

| Hodnota alpha | Chování |
|---|---|
| $\geq 10^{-2}$ | Pravděpodobná divergence (loss roste) |
| $10^{-3}$ | Rychlý pokles, ale nestabilní — může oscilovat |
| $10^{-4}$ | Stabilní konvergence (použité v cvičení) |
| $\leq 10^{-6}$ | Příliš pomalé — 500 kroků nestačí k dosažení dobrého výsledku |

Optimální hodnota závisí na: škálování vstupních dat, dimenzi prostoru ($d = 300$), počtu trénovacích párů a normách fastText vektorů. Pro robustnější trénink by bylo vhodné použít adaptivní optimizer (Adam, Adagrad), ale pro demonstraci principu je plain gradient descent s pevným `alpha` dostačující.

### 8.5 Proč loss neklesá na nulu

Ideální situace $L = 0$ by nastala, kdyby existovalo dokonalé lineární zobrazení z CZ do EN embedding prostoru. V realitě:

1. Embedding prostory různých jazyků nejsou perfektně izomorfní — lineární transformace je aproximace
2. Slovník obsahuje polysémní slova (jedno CS slovo → více EN překladů) — model musí kompromitovat
3. 500 kroků gradient descent s `alpha = 1e-4` nedosáhne ani lokálního minima — je to tradeoff rychlosti a přesnosti

Pro lepší výsledky by bylo možné:
- Trénovat déle (více kroků) nebo s adaptivním optimizerem
- Přidat regularizaci (penalizace velkých hodnot $W^T$)
- Použít ortogonální omezení na $W^T$ (MUSE toto dělá v pokročilé variantě)
- Zvýšit počet trénovacích párů
