package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// PostingEntry je jeden záznam v posting listu – dokument + četnost.
type PostingEntry struct {
	DocID     int
	Frequency int
}

// InvertedIndex mapuje token → posting list (seřazený podle DocID).
type InvertedIndex struct {
	Postings  map[string][]PostingEntry
	Documents []Document
}

// Česká stop slova pro filtraci při tokenizaci.
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
}

// tokenize normalizuje text na lowercase tokeny bez stop slov a jednoznakových slov.
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

// buildIndex vytvoří invertovaný index ze seznamu dokumentů.
func buildIndex(docs []Document) *InvertedIndex {
	idx := &InvertedIndex{
		Postings:  make(map[string][]PostingEntry),
		Documents: docs,
	}

	for _, doc := range docs {
		freq := make(map[string]int)
		for _, tok := range tokenize(doc.Title + " " + doc.Content) {
			freq[tok]++
		}
		for term, count := range freq {
			idx.Postings[term] = append(idx.Postings[term], PostingEntry{DocID: doc.ID, Frequency: count})
		}
	}

	for term := range idx.Postings {
		sort.Slice(idx.Postings[term], func(i, j int) bool {
			return idx.Postings[term][i].DocID < idx.Postings[term][j].DocID
		})
	}

	return idx
}

// lookup vrátí množinu DocID pro daný term.
func (idx *InvertedIndex) lookup(term string) map[int]struct{} {
	result := make(map[int]struct{})
	for _, e := range idx.Postings[strings.ToLower(term)] {
		result[e.DocID] = struct{}{}
	}
	return result
}

// allDocs vrátí množinu všech DocID v indexu.
func (idx *InvertedIndex) allDocs() map[int]struct{} {
	all := make(map[int]struct{}, len(idx.Documents))
	for _, d := range idx.Documents {
		all[d.ID] = struct{}{}
	}
	return all
}

// analyzeIndex vypíše statistiky o velikosti indexu.
func analyzeIndex(idx *InvertedIndex) {
	numTokens := len(idx.Postings)
	totalEntries := 0
	maxLen := 0
	maxTerm := ""
	for term, pl := range idx.Postings {
		totalEntries += len(pl)
		if len(pl) > maxLen {
			maxLen = len(pl)
			maxTerm = term
		}
	}
	avgLen := 0.0
	if numTokens > 0 {
		avgLen = float64(totalEntries) / float64(numTokens)
	}

	fmt.Println("\n=== Analýza indexu ===")
	fmt.Printf("Počet dokumentů:              %d\n", len(idx.Documents))
	fmt.Printf("Počet unikátních tokenů:      %d\n", numTokens)
	fmt.Printf("Celkový počet záznamů:        %d\n", totalEntries)
	fmt.Printf("Průměrná délka posting listu: %.2f\n", avgLen)
	fmt.Printf("Nejdelší posting list:        '%s' (%d dokumentů)\n", maxTerm, maxLen)
}
