package main

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
)

const (
	NumWords = 1000    // počet slov ve slovníku
	NumDocs  = 10000   // počet dokumentů
	NumPairs = 1000000 // počet náhodných unikátních dvojic (slovo, docID)
)

// InvertedIndex mapuje slovo → seřazený seznam docIDs.
type InvertedIndex map[string][]int

// EncodedIndex uchovává zakódované invertované seznamy pro jedno schéma.
type EncodedIndex struct {
	Name    string            // název schématu, např. "Unární"
	Encoded map[string]string // slovo → zakódovaný řetězec gapů
}

// generateWords vytvoří slovník NumWords náhodných slov délky 4–8 znaků.
func generateWords(rng *rand.Rand) []string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	words := make(map[string]struct{}, NumWords)
	result := make([]string, 0, NumWords)
	for len(result) < NumWords {
		length := 4 + rng.Intn(5) // 4–8 znaků
		b := make([]byte, length)
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		w := string(b)
		if _, exists := words[w]; !exists {
			words[w] = struct{}{}
			result = append(result, w)
		}
	}
	return result
}

// BuildInvertedIndex vygeneruje milion náhodných unikátních dvojic (slovo, docID)
// a sestaví invertovaný index — seřazený seznam docIDs pro každé slovo.
func BuildInvertedIndex(rng *rand.Rand, words []string) InvertedIndex {
	type pair struct {
		word  string
		docID int
	}
	seen := make(map[pair]struct{}, NumPairs)
	index := make(map[string]map[int]struct{}, NumWords)

	for len(seen) < NumPairs {
		w := words[rng.Intn(len(words))]
		d := 1 + rng.Intn(NumDocs) // docID 1..10000
		p := pair{w, d}
		if _, exists := seen[p]; !exists {
			seen[p] = struct{}{}
			if index[w] == nil {
				index[w] = make(map[int]struct{})
			}
			index[w][d] = struct{}{}
		}
	}

	inv := make(InvertedIndex, len(index))
	for w, docSet := range index {
		docs := make([]int, 0, len(docSet))
		for d := range docSet {
			docs = append(docs, d)
		}
		sort.Ints(docs)
		inv[w] = docs
	}
	return inv
}

// EncodeIndex zakóduje všechny invertované seznamy pomocí zadané encode funkce.
func EncodeIndex(name string, inv InvertedIndex, encodeFn func([]int) string) EncodedIndex {
	enc := EncodedIndex{Name: name, Encoded: make(map[string]string, len(inv))}
	for word, docs := range inv {
		gaps := ToGaps(docs)
		enc.Encoded[word] = encodeFn(gaps)
	}
	return enc
}

// VYHLEDÁVÁNÍ
// SearchRaw vyhledá docID v nativním invertovaném indexu (binární vyhledávání).
func SearchRaw(inv InvertedIndex, word string, docID int) bool {
	docs, ok := inv[word]
	if !ok {
		return false
	}
	pos := sort.SearchInts(docs, docID)
	return pos < len(docs) && docs[pos] == docID
}

// SearchEncoded vyhledá docID v zakódovaném indexu — dekóduje on-the-fly a porovnává.
func SearchEncoded(enc EncodedIndex, word string, docID int, decodeFn func(string) []int) bool {
	code, ok := enc.Encoded[word]
	if !ok {
		return false
	}
	gaps := decodeFn(code)
	docs := FromGaps(gaps)
	pos := sort.SearchInts(docs, docID)
	return pos < len(docs) && docs[pos] == docID
}

// velikosti

// RawSize vrátí velikost nezakódovaného invertovaného indexu v bajtech
// (jako textový zápis: "word:1 2 3\n").
func RawSize(inv InvertedIndex) int {
	total := 0
	for word, docs := range inv {
		nums := make([]string, len(docs))
		for i, d := range docs {
			nums[i] = fmt.Sprintf("%d", d)
		}
		total += len(word) + 1 + len(strings.Join(nums, " ")) + 1
	}
	return total
}

// EncodedSize vrátí celkovou velikost zakódovaného indexu v bajtech (délka bitových řetězců).
// Reálné binární uložení by bylo 8× menší (1 bit = 1 bajt v textové reprezentaci).
func EncodedSize(enc EncodedIndex) int {
	total := 0
	for _, code := range enc.Encoded {
		total += len(code)
	}
	return total
}

// EncodedSizeBytes přepočítá textovou délku bitů na ekvivalentní bajty (ceil(bits/8)).
func EncodedSizeBytes(enc EncodedIndex) int {
	bits := EncodedSize(enc)
	return (bits + 7) / 8
}
