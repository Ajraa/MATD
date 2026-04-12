package main

import (
	"strings"
	"unicode"
)

// stopWords obsahuje česká základní stopwords doplněná o vlastní výrazy
// (tedy, například, přičemž, zejména, přitom, avšak, nicméně, ovšem, ovšak, prostřednictvím).
var stopWords = map[string]struct{}{
	"a": {}, "ale": {}, "ani": {}, "aby": {}, "bez": {}, "co": {}, "či": {},
	"do": {}, "ho": {}, "i": {}, "je": {}, "jej": {}, "jeho": {}, "její": {},
	"jejich": {}, "ji": {}, "již": {}, "jak": {}, "jako": {}, "jsou": {},
	"jsem": {}, "jste": {}, "jsme": {}, "jen": {}, "k": {}, "ke": {},
	"kde": {}, "kdo": {}, "když": {}, "má": {}, "mám": {}, "mi": {},
	"mít": {}, "mně": {}, "mu": {}, "na": {}, "nad": {}, "nebo": {},
	"ní": {}, "no": {}, "o": {}, "od": {}, "pak": {}, "po": {}, "pod": {},
	"podle": {}, "pro": {}, "proto": {}, "při": {}, "před": {}, "přes": {},
	"s": {}, "se": {}, "si": {}, "ta": {}, "tak": {}, "také": {}, "tam": {},
	"tato": {}, "te": {}, "tě": {}, "ten": {}, "tento": {}, "tím": {},
	"to": {}, "tomu": {}, "toto": {}, "tu": {}, "ty": {}, "u": {},
	"v": {}, "ve": {}, "z": {}, "za": {}, "ze": {}, "zde": {}, "že": {},
	"byl": {}, "byla": {}, "bylo": {}, "byly": {}, "bude": {}, "být": {},
	"by": {}, "bych": {}, "bychom": {}, "byste": {}, "byli": {}, "budou": {},
	"ne": {}, "není": {}, "nejsou": {}, "ještě": {}, "jenom": {}, "mě": {},
	"nás": {}, "nám": {}, "náš": {}, "naše": {}, "svých": {}, "těch": {},
	"více": {}, "vše": {}, "všech": {}, "všechny": {}, "vám": {}, "váš": {},
	"vaše": {}, "vás": {}, "budu": {}, "budeš": {}, "budeme": {}, "budete": {},
	"an": {}, "sv": {}, "them": {}, "the": {}, "this": {},
	"tedy": {}, "například": {}, "přičemž": {}, "zejména": {}, "přitom": {},
	"avšak": {}, "nicméně": {}, "ovšem": {}, "ovšak": {}, "prostřednictvím": {},
}

// tokenize normalizuje text na lowercase tokeny bez stop slov a jednoznakových slov.
// Odstraňuje interpunkci (vše co není IsLetter nebo IsDigit), filtruje stopwords.
func tokenize(text string) []string {
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	result := tokens[:0]
	for _, t := range tokens {
		if _, isStop := stopWords[t]; !isStop && len([]rune(t)) > 1 {
			result = append(result, t)
		}
	}
	return result
}

// preprocessCorpus předzpracuje všechny dokumenty a vrátí mapu DocID → seznam termů.
func preprocessCorpus(docs []Document) map[int][]string {
	result := make(map[int][]string, len(docs))
	for _, doc := range docs {
		result[doc.ID] = tokenize(doc.Title + " " + doc.Content)
	}
	return result
}
