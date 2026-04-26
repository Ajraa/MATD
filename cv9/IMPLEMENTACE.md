# CV9 — Implementace CBOW modelu pro češtinu pomocí Keras

## Obsah

1. [Úvod a cíl](#1-úvod-a-cíl)
2. [Teoretický základ](#2-teoretický-základ)
3. [Část 1 — Příprava dat](#3-část-1--příprava-dat)
4. [Část 2 — Odvození gradientů](#4-část-2--odvození-gradientů)
5. [Část 3 — Architektura modelu a trénování](#5-část-3--architektura-modelu-a-trénování)
6. [Část 4 — Vyhodnocení modelu](#6-část-4--vyhodnocení-modelu)
7. [Tok dat — end-to-end přehled](#7-tok-dat--end-to-end-přehled)
8. [Klíčové implementační detaily a možná úskalí](#8-klíčové-implementační-detaily-a-možná-úskalí)

---

## 1. Úvod a cíl

### Problém

Distribuované vektorové reprezentace slov (embeddingy) jsou základním stavebním kamenem moderního zpracování přirozeného jazyka. Cílem tohoto cvičení je od základů pochopit a implementovat jeden ze základních modelů, který tyto reprezentace vytváří — **CBOW (Continuous Bag of Words)**.

Na rozdíl od cvičení 8, kde byly použity hotové fastText embeddingy, cvičení 9 trénuje vlastní embeddingy na českém korpusu. Model se učí, jaký sémantický kontext obklopuje každé slovo, a tuto informaci zakóduje do hustých vektorů.

### Metoda CBOW (Word2Vec)

CBOW je jednou ze dvou architektur rodiny **Word2Vec** (Mikolov et al., 2013). Princip je inverzní k modelu Skip-gram:

- **Skip-gram**: Ze středového slova předpovídá kontext.
- **CBOW**: Z kontextu (okolních slov) předpovídá středové slovo.

CBOW je výpočetně efektivnější — model nemusí řešit výstup pro každé kontextové slovo zvlášť, ale produkuje jednu předpověď na jedno okno.

### Co je výsledkem

Natrénovaná embedding matice $E \in \mathbb{R}^{|V| \times d}$, kde:

- $|V| = 20\,000$ — velikost slovníku (nejčastějších slov z korpusu)
- $d = 100$ — dimenze embeddingů

Po 5 epochách tréninku na ~37 M znacích (5,2 M tokenů) českého textu dosahuje implementace:

- **Trénovací přesnost: ~20 %** po první epoše, ~20 % po páté epoše
- **Validační přesnost: ~18 %** po první epoše, ~22 % po páté epoše

Nízká absolutní přesnost je u CBOW standardní — predikce jednoho slova z 20 000 je extrémně obtížná úloha. Důležitá není přesnost samotná, ale **kvalita výsledných embeddingů** — jejich schopnost zachytit sémantické vztahy.

---

## 2. Teoretický základ

### 2.1 Co je CBOW

**CBOW (Continuous Bag of Words)** je neuronová síť s jednou skrytou vrstvou (embedding vrstvou). Model dostane indexy $2k$ kontextových slov (okno $k$ vlevo + $k$ vpravo od středového slova) a naučí se předpovídat index středového slova.

Klíčové vlastnosti:
- Kontextová okna jsou **bag of words** — na pořadí slov nezáleží, pracuje se pouze s průměrem embeddingů.
- Parametry modelu jsou dvě matice: vstupní embedding matice $E$ a výstupní váhová matice $W$.
- Po tréninku se jako výsledné embeddingy používají **řádky matice $E$** — každý řádek je vektorem jednoho slova.

### 2.2 Architektura

Model se skládá ze čtyř kroků:

**1. Lookup** — každé z $2k$ kontextových slov je namapováno na vektor dimenze $d$:

$$E[c_i] \in \mathbb{R}^d, \quad i = 1, \ldots, 2k$$

**2. Průměrování** — embeddingy kontextových slov se zprůměrují:

$$v_\text{ctx} = \frac{1}{2k} \sum_{i=1}^{2k} E[c_i] \in \mathbb{R}^d$$

**3. Lineární projekce + softmax** — výstupní vrstva transformuje $v_\text{ctx}$ na pravděpodobnostní distribuci přes slovník:

$$z = W^\top v_\text{ctx} + b \in \mathbb{R}^{|V|}$$

$$\hat{y}_j = \text{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{|V|} e^{z_k}}$$

**4. Ztráta** — kategoriální křížová entropie vůči one-hot reprezentaci cílového slova $c$:

$$L = -\log \hat{y}_c = -z_c + \log \sum_{k=1}^{|V|} e^{z_k}$$

### 2.3 Word2Vec a distributional hypothesis

Word2Vec je inspirován **distributional hypothesis** (Firth, 1957): *slova, která se vyskytují ve stejných kontextech, mají podobný význam.* CBOW tuto hypotézu operacionalizuje přímo do tréninkového signálu — model se naučí podobné vektory pro slova, která mají podobné okolí v textu.

### 2.4 Proč průměrování kontextů

CBOW průměruje embeddingy všech kontextových slov do jednoho vektoru. Výhody:
- Redukce dimenzionality při zachování sémantické informace
- Lineární operace — jednoduchý gradient (viz Část 2)
- Robustnost vůči pořadí slov a šumu v kontextu

Nevýhoda: průměrování ztrácí informaci o pozici a pořadí slov — proto CBOW nepracuje dobře pro úlohy, kde syntaktická pozice záleží (skip-gram je na tyto jevy citlivější).

### 2.5 Kosinová podobnost a hodnocení embeddingů

Pro vyhodnocení embeddingů se používá kosinová podobnost (stejný důvod jako v CV8 — délka vektoru nezáleží, pouze směr nese sémantický obsah):

$$\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

Nejbližší sousedé slova jsou slova s nejvyšší kosinovou podobností — ideálně by to měla být sémanticky příbuzná slova.

---

## 3. Část 1 — Příprava dat

### 3.1 Konfigurace

```python
CORPUS_FILE   = 'cs.txt'    # Cesta ke korpusu
VOCAB_SIZE    = 20_000      # Počet nejčastějších slov ve slovníku
CONTEXT_SIZE  = 2           # Okno kontextu (2 vlevo + 2 vpravo)
EMBED_DIM     = 100         # Dimenze embeddingů
BATCH_SIZE    = 512
EPOCHS        = 5
UNK_TOKEN     = '<UNK>'
MODEL_FILE    = 'cbow_model.keras'
VOCAB_FILE    = 'vocab.pkl'
```

Volby konfiguračních konstant:
- `VOCAB_SIZE = 20 000` — dostatečně velký slovník pro zachycení běžné slovní zásoby, ale dostatečně malý pro efektivní softmax (větší slovník = větší výstupní vrstva)
- `CONTEXT_SIZE = 2` — standardní hodnota z původního Word2Vec článku; okno ±2 zachytí bezprostřední kontext bez zahrnutí nesouvisejících slov
- `EMBED_DIM = 100` — kompromis: fastText používá 300, ale 100 je dostačující pro demonstraci principu a trénink je 3× rychlejší
- `BATCH_SIZE = 512` — větší batch = stabilnější gradient, rychlejší trénink na GPU/CPU

### 3.2 `tokenize` — tokenizace textu

```python
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'[a-záčďéěíňóřšťúůýž]+', text)
    return tokens
```

Tokenizace je záměrně jednoduchá:

1. **Převod na malá písmena** — "Praha" a "Praha" jsou stejné slovo; case-sensitive model by zbytečně fragmentoval slovník
2. **Extrakce slov** — regulární výraz `[a-záčďéěíňóřšťúůýž]+` zachytí pouze slova z české abecedy (včetně diakritiky), ignoruje čísla, interpunkci a speciální znaky

**Výsledek na korpusu:**
- Načteno 37 655 506 znaků
- Celkem 5 201 340 tokenů (průměrně ~7,2 znaku na token)

### 3.3 Sestavení slovníku

```python
counter = Counter(tokens)
most_common = counter.most_common(VOCAB_SIZE - 1)

word2idx = {UNK_TOKEN: 0}
for word, _ in most_common:
    word2idx[word] = len(word2idx)
idx2word = {v: k for k, v in word2idx.items()}
```

**Postup:**
1. Spočítá frekvenci všech slov v korpusu pomocí `Counter`
2. Vezme `VOCAB_SIZE - 1` nejčastějších slov (místo rezervuje pro `<UNK>`)
3. `<UNK>` dostane index 0 — konvence pro neznámá slova
4. Ostatní slova dostávají indexy 1 až `VOCAB_SIZE - 1` v pořadí od nejčastějšího

**Proč index 0 pro `<UNK>`?** Konvence umožňuje jednoduché fallback: `word2idx.get(token, 0)` vrátí 0 pro jakékoli neznámé slovo.

**Top 10 slov ve slovníku:**

| Pořadí | Slovo | Frekvence |
|---|---|---|
| 1 | a | 166 767 |
| 2 | v | 160 790 |
| 3 | na | 78 895 |
| 4 | se | 78 136 |
| 5 | je | 50 300 |
| 6 | s | 45 733 |
| 7 | z | 41 243 |
| 8 | roce | 34 012 |
| 9 | do | 30 646 |
| 10 | byl | 30 118 |

Dominance krátkých funkčních slov je typická pro češtinu — oproti angličtině mají předložky a spojky vysokou frekvenci.

### 3.4 Kódování tokenu na indexy

```python
encoded = [word2idx.get(t, 0) for t in tokens]
unk_ratio = encoded.count(0) / len(encoded)
```

Výsledný `unk_ratio = 22.18 %` — téměř čtvrtina tokenů padne do `<UNK>`. To je relativně vysoké číslo. Příčiny:
- Česká morfologie je bohatá — jedno slovo má desítky tvarů (skloňování, časování), a i s 20 000 slovními tvary ve slovníku zůstane mnoho vzácných forem jako `<UNK>`
- Slovník obsahuje jen 19 999 nejčastějších tvarů (nikoli lemmat)

### 3.5 `build_training_data` — generování trénovacích dvojic

```python
def build_training_data(encoded_tokens, context_size):
    contexts = []
    targets  = []
    n = len(encoded_tokens)
    for i in range(context_size, n - context_size):
        target = encoded_tokens[i]
        if target == 0:       # Přeskočit <UNK> jako cíl
            continue
        ctx = (
            encoded_tokens[i - context_size : i] +
            encoded_tokens[i + 1 : i + context_size + 1]
        )
        contexts.append(ctx)
        targets.append(target)
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)
```

**Princip posuvného okna:**

Pro každou pozici $i$ v textu (s výjimkou krajů) se sestaví dvojice:

- **Kontext:** $2k$ indexů slov v okolí: $[i-k, \ldots, i-1, i+1, \ldots, i+k]$
- **Cíl:** index středového slova $i$

**Proč přeskakovat `<UNK>` jako cíl?**

Predikce `<UNK>` (index 0) by model nic nenaučila — `<UNK>` reprezentuje tisíce různých vzácných slov bez jakéhokoliv konzistentního sémantického obsahu. Zahrnutím `<UNK>` do cílů by model trénoval chybný signál a embedding pro `<UNK>` by byl nesourodý. Kontextová slova naopak mohou být `<UNK>` — ovlivňují průměr jen málo a jejich index 0 je validní vstup do embedding vrstvy.

**Výsledek:**

- Trénovacích dvojic: 4 047 505
- Tvar `X`: `(4 047 505, 4)` — čtyři kontextové indexy (2 vlevo + 2 vpravo)
- Tvar `y`: `(4 047 505,)` — jeden cílový index

---

## 4. Část 2 — Odvození gradientů

Tato sekce odpovídá teoretické části cvičení. Odvozuje gradienty pro ručně implementovaný backpropagation CBOW modelu.

### 4.1 Gradient podle vstupu do softmaxu ($\partial L / \partial z$)

Označme:
- $z \in \mathbb{R}^{|V|}$ — logity (vstup do softmaxu)
- $\hat{y}_j = e^{z_j} / \sum_k e^{z_k}$ — výstup softmaxu
- $c$ — index správného (cílového) slova, $y_c = 1$, ostatní $y_j = 0$

Loss (křížová entropie):

$$L = -\log \hat{y}_c = -z_c + \log \sum_k e^{z_k}$$

Derivace:

$$\frac{\partial L}{\partial z_j} = -\mathbf{1}[j=c] + \frac{e^{z_j}}{\sum_k e^{z_k}} = \hat{y}_j - y_j$$

Vektorově:

$$\boxed{\frac{\partial L}{\partial z} = \hat{y} - y_\text{one-hot}}$$

**Interpretace:** Gradient je rozdíl mezi predikovanou distribucí $\hat{y}$ a správnou distribucí $y$. Pro správnou třídu je $\hat{y}_c - 1$ (záporné — chceme zvýšit $z_c$). Pro všechny ostatní třídy je $\hat{y}_j$ (kladné — chceme snížit jejich logity).

### 4.2 Gradient podle výstupních vah ($\partial L / \partial W$)

Výstupní vrstva: $z = W^\top v_\text{ctx} + b$, kde $W \in \mathbb{R}^{d \times |V|}$.

Řetězové pravidlo:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W} = v_\text{ctx} \cdot (\hat{y} - y)^\top$$

**Kontrola rozměrů:** $v_\text{ctx} \in \mathbb{R}^{d \times 1}$, $(\hat{y} - y)^\top \in \mathbb{R}^{1 \times |V|}$ → výsledek $\in \mathbb{R}^{d \times |V|}$ — shoduje se s rozměry $W$. ✓

$$\boxed{\frac{\partial L}{\partial W} = v_\text{ctx} (\hat{y} - y)^\top}$$

### 4.3 Gradient podle kontextového vektoru ($\partial L / \partial v_\text{ctx}$)

Z $z = W^\top v_\text{ctx}$ plyne přímou aplikací pravidla pro maticové násobení:

$$\frac{\partial L}{\partial v_\text{ctx}} = W \cdot (\hat{y} - y)$$

**Kontrola rozměrů:** $W \in \mathbb{R}^{d \times |V|}$, $(\hat{y} - y) \in \mathbb{R}^{|V|}$ → výsledek $\in \mathbb{R}^d$. ✓

$$\boxed{\frac{\partial L}{\partial v_\text{ctx}} = W(\hat{y} - y)}$$

### 4.4 Distribuce gradientu na embeddingy

V CBOW je $v_\text{ctx} = \frac{1}{2k} \sum_{i=1}^{2k} E[c_i]$. Průměrování je lineární operace, takže:

$$\frac{\partial v_\text{ctx}}{\partial E[c_i]} = \frac{1}{2k} \cdot \mathbf{I}$$

Řetězovým pravidlem:

$$\frac{\partial L}{\partial E[c_i]} = \frac{\partial L}{\partial v_\text{ctx}} \cdot \frac{\partial v_\text{ctx}}{\partial E[c_i]} = \frac{1}{2k} \cdot W(\hat{y} - y)$$

**Klíčové pozorování:** Gradient se rozdělí **rovnoměrně** na všechna $2k$ kontextová slova, bez ohledu na jejich pozici nebo frekvenci. To je přímý důsledek symetrie operace průměrování — každé slovo přispívá do $v_\text{ctx}$ stejným dílem.

**Praktický dopad:** Embeddingy frekventovaných kontextových slov (a, v, na, se…) se aktualizují při každém kroku, embeddingy vzácných slov málokdy. Frekventovaná slova proto konvergují rychleji, vzácná pomaleji.

---

## 5. Část 3 — Architektura modelu a trénování

### 5.1 `build_cbow_model` — Keras model

```python
def build_cbow_model(vocab_size, embed_dim, context_size):
    inputs = keras.Input(shape=(2 * context_size,), name='context')
    emb = layers.Embedding(vocab_size, embed_dim, name='embeddings')(inputs)
    avg = layers.Lambda(lambda x: keras.ops.mean(x, axis=1), name='avg_pool')(emb)
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(avg)
    model = Model(inputs, outputs, name='CBOW')
    return model
```

**Vrstva po vrstvě:**

| Vrstva | Typ | Vstup → Výstup | Popis |
|---|---|---|---|
| `context` | `InputLayer` | – → `(None, 4)` | Vstup: 4 kontextové indexy |
| `embeddings` | `Embedding` | `(None, 4)` → `(None, 4, 100)` | Lookup: každý index → 100-dim vektor |
| `avg_pool` | `Lambda` | `(None, 4, 100)` → `(None, 100)` | Průměr přes 4 vektory (axis=1) |
| `output` | `Dense` | `(None, 100)` → `(None, 20000)` | Lineární projekce + softmax |

**Celkem parametrů: 4 020 000 (15,34 MB)**

| Složka | Parametry |
|---|---|
| Embedding matice $E$ | $20\,000 \times 100 = 2\,000\,000$ |
| Výstupní váhy $W$ | $100 \times 20\,000 = 2\,000\,000$ |
| Výstupní bias $b$ | $20\,000$ |

**Proč `Lambda` místo `GlobalAveragePooling1D`?** Funkčně jsou ekvivalentní — obě průměrují přes osu 1. `Lambda` s `keras.ops.mean` je explicitnější a demonstrativnější.

### 5.2 Kompilace

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

**Volba optimizeru — Adam:** Na rozdíl od CV8, kde byl použit plain gradient descent, Keras model využívá **Adam** (Adaptive Moment Estimation). Adam kombinuje:
- **Momentum** — pohyb ve směru průměrného gradientu (vyhlazení oscilací)
- **Adaptivní learning rate** — každý parametr má vlastní efektivní `alpha` přizpůsobené jeho historii gradientů

Pro embedding modely je Adam výrazně efektivnější než plain SGD — frekventovaná slova dostávají malé kroky (jsou dobře odhadnutá), vzácná slova větší kroky (potřebují intenzivnější update).

**`sparse_categorical_crossentropy` vs `categorical_crossentropy`:** Pro velké slovníky ($|V| = 20\,000$) je klíčová volba `sparse_` varianty — přijímá **indexy** jako cíle (skalár), nikoli one-hot vektory. One-hot vektor délky 20 000 by pro 4 miliony příkladů spotřeboval ~300 GB RAM.

### 5.3 Trénink

```python
history = model.fit(
    X, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.05,
    verbose=1,
)
```

**Průběh tréninku:**

| Epocha | Trénovací loss | Trénovací accuracy | Validační loss | Validační accuracy |
|---|---|---|---|---|
| 1 | 7,3268 | 10,27 % | 6,4432 | 18,31 % |
| 2 | 6,3794 | 16,10 % | 6,0404 | 20,36 % |
| 3 | 5,9766 | 17,99 % | 5,8430 | 21,38 % |
| 4 | 5,7225 | 19,15 % | 5,7273 | 21,87 % |
| 5 | 5,5419 | 20,00 % | 5,6589 | 22,16 % |

**Proč je validační accuracy vyšší než trénovací?**

Tento jev je překvapivý, ale má snadné vysvětlení: `validation_split=0.05` oddělí posledních 5 % dat jako validační sadu. V textovém datasetu jsou to konkrétní slova z konce souboru. Jelikož model vidí trénovací data v každé epoše postupně, validační data jsou vyhodnocena vždy po celé epoše — v té chvíli jsou váhy již plně natrénované. Naopak trénovací accuracy je průměr přes celou epochu (včetně začátku, kdy jsou váhy slabší). Výsledkem je zdánlivě vyšší validační přesnost.

**Délka tréninku:** Každá epocha trvala ~670-720 sekund (CPU), celkem ~55 minut. GPU by toto zkrátilo na minuty.

### 5.4 Extrakce embeddingů

```python
model.save(MODEL_FILE)
embedding_matrix = model.get_layer('embeddings').get_weights()[0]
# embedding_matrix.shape = (20000, 100)
```

Embeddingy jsou uloženy jako váhy `Embedding` vrstvy — matice $(20\,000, 100)$, kde řádek $i$ je 100-dimenzionální vektor slova s indexem $i$.

---

## 6. Část 4 — Vyhodnocení modelu

### 6.1 Nejbližší sousedé (kosinová podobnost)

```python
def get_embedding(word):
    idx = word2idx.get(word, 0)
    return embedding_matrix[idx]

def nearest_neighbors(word, top_n=5):
    if word not in word2idx:
        return []
    vec = get_embedding(word).reshape(1, -1)
    sims = cosine_similarity(vec, embedding_matrix)[0]
    sims[word2idx[word]] = -1  # Vyloučit samotné slovo
    top_idxs = np.argsort(sims)[::-1][:top_n]
    return [(idx2word[i], float(sims[i])) for i in top_idxs]
```

**Postup:**

1. Získá embedding testovaného slova — vektor $(1, 100)$
2. `cosine_similarity(vec, embedding_matrix)` spočítá podobnost vůči všem 20 000 slovům najednou — maticová operace $(1, 100) \cdot (20000, 100)^\top = (1, 20000)$
3. Vlastní slovo se vyloučí nastavením jeho podobnosti na $-1$
4. `np.argsort` seřadí od největší podobnosti

**Výsledky na testovacích slovech:**

| Testované slovo | Top 5 nejbližších sousedů |
|---|---|
| pes | klasický (0.631), čaj (0.614), živý (0.587), zelený (0.569), náš (0.549) |
| škola | knihovna (0.674), pec (0.661), univerzita (0.654), soukromá (0.642), technická (0.626) |
| krásný | kruhový (0.694), malý (0.690), plochý (0.683), nahoře (0.656), modrý (0.650) |
| město | obec (0.624), nádraží (0.570), knížectví (0.549), městečko (0.525), městys (0.522) |
| věda | ekonomická (0.694), kultura (0.658), odborná (0.655), forma (0.647), matematika (0.643) |

**Hodnocení výsledků:**
- **"město" → "obec", "městečko", "městys"** — výborný výsledek, model zachytil správné sémantické příbuzné
- **"škola" → "knihovna", "univerzita"** — smysluplné asociace (vzdělávací instituce)
- **"pes" → "klasický", "čaj"** — slabý výsledek; model pravděpodobně neměl dostatek dat s výskytem slova "pes" v konzistentním kontextu
- Výsledky by se výrazně zlepšily s větším korpusem a více epochami

### 6.2 Vizualizace embedding prostoru (t-SNE + PCA)

Pro vizualizaci se použijí 2 000 nejčastějších slov:

```python
N_VIS = 2000
vis_vecs = embedding_matrix[vis_idxs]  # (2000, 100)

# t-SNE: nelineární redukce dimenzionality
tsne = TSNE(n_components=2, random_state=42, perplexity=40)
vis_2d = tsne.fit_transform(vis_vecs)

# PCA: lineární redukce dimenzionality
pca = PCA(n_components=2, random_state=42)
pca_2d = pca.fit_transform(vis_vecs)
```

**t-SNE vs PCA:**

| Metoda | Typ | Zachovává | Vhodné pro |
|---|---|---|---|
| **t-SNE** | Nelineární | Lokální sousedství (clustery) | Vizualizaci shluků, skupin příbuzných slov |
| **PCA** | Lineární | Globální varianci (první komponenty) | Rychlý přehled, interpretovatelné osy |

Oba grafy zvýrazní slova jako "pes", "kočka", "škola", "učitel", "muž", "žena", "král", "Praha", "Brno" červenou barvou s popiskem. Zbývající slova jsou zobrazena jako šedé body.

**Co t-SNE/PCA ukazuje:**
- Slova se podobným kontextem by měla tvořit shluky (geografické názvy, profese, přídavná jména)
- PCA zachytí hlavní dimenze variance — v embedding prostoru to může odpovídat sémantickým osám (gender, konkrétnost/abstraktnost)

### 6.3 Test analogií

```python
def analogy(a, b, c, top_n=5, exclude_input=True):
    """
    Analogie: a : b = c : ?
    Hledá slova nejbližší vektoru v(b) - v(a) + v(c).
    """
    query_vec = get_embedding(b) - get_embedding(a) + get_embedding(c)
    query_vec = query_vec.reshape(1, -1)
    sims = cosine_similarity(query_vec, embedding_matrix)[0]
    if exclude_input:
        for w in (a, b, c):
            sims[word2idx[w]] = -1
    top_idxs = np.argsort(sims)[::-1][:top_n]
    return [(idx2word[i], float(sims[i])) for i in top_idxs]
```

**Princip vektorové analogie:**

Pokud embeddingy zachycují sémantické vztahy, mělo by platit:

$$\vec{b} - \vec{a} + \vec{c} \approx \vec{d}$$

Pro analogii $a : b = c : d$ (např. "muž : žena = král : královna").

**Výsledky testovacích analogií:**

| Analogie | Očekáváno | Top výsledek | Zhodnocení |
|---|---|---|---|
| muž : žena = král : ? | královna | manželka (0.707) | Sémanticky blízké (ženský rod + autorita), ale ne přesné |
| Praha : česko = Paříž : ? | Francie | **francie (0.615)** | Správná odpověď na 1. místě! |
| pes : psi = kočka : ? | kočky | algebry (0.580) | Nesprávný výsledek — morfologické vztahy slabě zachyceny |

**Proč analogie nefungují perfektně?**

1. **Malý korpus** — 5 M tokenů je řádově méně než Google News (100 mld slov) nebo Wikipedia (2 mld slov), na nichž Word2Vec dosahuje vynikajících výsledků
2. **Málo epoch** — 5 epoch nestačí k úplné konvergenci na velkém slovníku
3. **Malá dimenzionalita** — 100 dimenzí vs standardních 300 při Word2Vec

### 6.4 Analýza genderového biasu

```python
bias_tests = [
    ('muž', 'žena', 'doktor',   'doktorka'),
    ('muž', 'žena', 'inženýr',  'inženýrka'),
    ('muž', 'žena', 'vědec',    'vědkyně'),
]
```

**Výsledky:**

| Analogie | Očekáváno | Top výsledek |
|---|---|---|
| muž : žena = doktor : ? | doktorka | manželka (0.683) |
| muž : žena = inženýr : ? | inženýrka | básnířka (0.631) |
| muž : žena = vědec : ? | vědkyně | kritička (0.685) |

**Interpretace:**

Model konzistentně vrací ženská slova (ženské přípony `-ka`, `-ice`, `-yně`), ale ne nutně profese odpovídající vstupu. Výsledky jako "manželka" pro "doktorka" mohou odrážet:

- **Jazykový bias v korpusu** — slova "žena" a "manželka" se v textu vyskytují ve velmi podobném kontextu
- **Nedostatek trénovacích dat** — vzácné profese (doktorka, inženýrka) mají málo výskytů, jejich embeddingy jsou nepřesné

Správná odezva (femininum profesní názvy) by vyžadovala buď větší korpus, nebo jiný přístup (lemmatizace + morfologický slovník).

---

## 7. Tok dat — end-to-end přehled

```
Korpus (cs.txt)
═══════════════
37 655 506 znaků
        │
        ▼
   tokenize()
        │
5 201 340 tokenů (lowercase, jen česká slova)
['jan', 'tříska', 'listopadu', 'Praha', ...]
        │
        ▼
   Counter() + most_common(19999)
        │
   word2idx  (20 000 slov, <UNK>=0)
   idx2word  (inverzní mapa)
        │
        ▼
   encode: token → index (OOV → 0)
        │
   encoded  (5 201 340 int32, UNK_ratio=22.18 %)
        │
        ▼
   build_training_data(context_size=2)
   ┌────────────────────────────────────────────────────────┐
   │  for i in range(2, n-2):                               │
   │    if encoded[i] == 0: continue                        │
   │    ctx = encoded[i-2:i] + encoded[i+1:i+3]             │
   │    contexts.append(ctx)  targets.append(encoded[i])    │
   └────────────────────────────────────────────────────────┘
        │
   X  (4 047 505, 4)    — kontextové indexy
   y  (4 047 505,)      — cílové indexy
        │
        ▼
   build_cbow_model(vocab=20000, dim=100, ctx=2)
   ┌────────────────────────────────────────────────────────┐
   │  Input(4) → Embedding(20000,100) → Lambda(mean,axis=1) │
   │          → Dense(20000, softmax)                       │
   │  Adam(lr=0.001), sparse_categorical_crossentropy       │
   └────────────────────────────────────────────────────────┘
        │
   model.fit(X, y, batch=512, epochs=5, val_split=0.05)
        │
   Epoch 1: loss=7.33, acc=10.3 %
   Epoch 5: loss=5.54, acc=20.0 %
        │
        ▼
   embedding_matrix = model.get_layer('embeddings').get_weights()[0]
   embedding_matrix.shape = (20000, 100)
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
nearest_neighbors("město")                  analogy("muž","žena","král")
        │                                              │
cosine_similarity(E["město"], E)          v(žena)-v(muž)+v(král) → top-5
        │                                              │
["obec", "nádraží", ...]                  ["manželka", "matka", ...]
        │
        ▼
t-SNE / PCA vizualizace 2000 slov
```

---

## 8. Klíčové implementační detaily a možná úskalí

### 8.1 Softmax nad velkým slovníkem je úzké hrdlo

Výstupní vrstva `Dense(20000, activation='softmax')` musí při každém forward passu:

1. Spočítat logity: $(N, 100) \cdot (100, 20000) = (N, 20000)$ — $N \cdot 100 \cdot 20000 = 2\times10^9$ operací pro batch 512
2. Aplikovat softmax přes 20 000 hodnot

Pro velký slovník (100 000+) by se v produkci použil **negative sampling** nebo **hierarchical softmax** — obě metody aproximují softmax a drasticky snižují výpočetní náklady. Cvičení používá plný softmax jako didakticky jednodušší variantu.

### 8.2 Proč `<UNK>` nelze predikovat

V `build_training_data` jsou přeskočeny všechny pozice, kde je středové slovo `<UNK>` (index 0). Důvod je fundamentální: `<UNK>` reprezentuje tisíce různých vzácných slov s navzájem nesouvisejícím kontextem. Trénink na predikci `<UNK>` by model aktivně škodil — embedding vektoru 0 by byl průměrem všech vzácných slov, tedy sémanticky nesmyslný, a backpropagace by vnášela šum do všech okolních embeddingů.

### 8.3 `validation_split` odděluje konec souboru

`validation_split=0.05` v Keras vždy vezme posledních 5 % vzorků (nepromíchá). V textovém datasetu to znamená, že validační sada pochází z konce souboru. Pro Wikipedii nebo beletrii to obvykle nepřináší problém (text je dostatečně heterogenní), ale pro specializované texty by mohl být konec souboru tematicky jiný než začátek.

### 8.4 Proč výsledky analogií závisí na velikosti korpusu

Word2Vec analogie fungují, protože opakovaná koexpozice slov v konzistentním kontextu vytvoří geometricky pravidelné vztahy. S malým korpusem:

- Vzácné páry (muž/žena-profese) mají málo koexpozic → nepřesné embeddingy
- Frekventovaná funkční slova (a, v, na) dominují kontextovým oknům a "zahlcují" signal
- Morfologické varianty (pes/psovi/psu) mají separátní embeddingy → vztahy mezi tvary jsou slabé

Řešení: lemmatizace před trénováním by sjednotila tvary jednoho slova a zkvalitnila embeddingy.

### 8.5 Citlivost t-SNE na parametr `perplexity`

```python
tsne = TSNE(n_components=2, random_state=42, perplexity=40)
```

`perplexity` (zde 40) kontroluje efektivní počet sousedů, které t-SNE bere v úvahu. Doporučené hodnoty jsou 5–50. Pro 2 000 bodů je 40 rozumná volba:
- Příliš malá perplexity → fragmentované ostrůvky bez struktury
- Příliš velká perplexity → celý prostor se zploští do jedné koule

t-SNE **není deterministický** při různých hodnotách `perplexity` nebo `random_state` — výsledné rozmístění bodů se mění, ale topologické clustery (skupiny příbuzných slov) by měly zůstat.

### 8.6 GPU paměť a `set_memory_growth`

```python
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

Bez `set_memory_growth` by TensorFlow alokoval **veškerou dostupnou VRAM** při startu. Pro sdílené GPU (cluster, Colab) nebo víceúlohové prostředí je to nežádoucí. `set_memory_growth=True` způsobí, že TensorFlow alokuje paměť postupně podle potřeby.
