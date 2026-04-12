package main

import (
	"math"
	"sort"
)

// DocPair reprezentuje dvojici dokumentů s jejich kosinovou podobností.
type DocPair struct {
	DocA       int
	DocB       int
	Similarity float64
}

// cosineSimilarity vypočítá kosinovou podobnost dvou vektorů reprezentovaných jako mapy term→váha.
// Vrátí 0, pokud je norma jednoho z vektorů nulová.
func cosineSimilarity(v1, v2 map[string]float64) float64 {
	// Skalární součin (dot product) – iterujeme přes menší vektor
	dot := 0.0
	for term, w1 := range v1 {
		if w2, ok := v2[term]; ok {
			dot += w1 * w2
		}
	}

	// Normy obou vektorů
	norm1 := 0.0
	for _, w := range v1 {
		norm1 += w * w
	}
	norm1 = math.Sqrt(norm1)

	norm2 := 0.0
	for _, w := range v2 {
		norm2 += w * w
	}
	norm2 = math.Sqrt(norm2)

	// Ochrana před dělením nulou
	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dot / (norm1 * norm2)
}

// allPairsSimilarity vrátí všechny dvojice dokumentů seřazené sestupně podle kosinové
// podobnosti TF-IDF vektorů.
func allPairsSimilarity(idx *TFIDFIndex) []DocPair {
	return computePairs(idx.TFIDF)
}

// allPairsSimilarityTFOnly vrátí všechny dvojice dokumentů seřazené sestupně podle kosinové
// podobnosti čistých TF vektorů (bez IDF) – pro porovnání s TF-IDF variantou.
func allPairsSimilarityTFOnly(idx *TFIDFIndex) []DocPair {
	return computePairs(idx.TF)
}

// computePairs je pomocná funkce, která spočítá všechny dvojice dokumentů
// na základě předaných vektorů a seřadí je sestupně podle podobnosti.
func computePairs(vectors map[int]map[string]float64) []DocPair {
	// Seřazená lista DocID pro deterministické pořadí
	docIDs := make([]int, 0, len(vectors))
	for id := range vectors {
		docIDs = append(docIDs, id)
	}
	sort.Ints(docIDs)

	n := len(docIDs)
	pairs := make([]DocPair, 0, n*(n-1)/2)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a := docIDs[i]
			b := docIDs[j]
			sim := cosineSimilarity(vectors[a], vectors[b])
			pairs = append(pairs, DocPair{DocA: a, DocB: b, Similarity: sim})
		}
	}

	// Seřazení sestupně podle podobnosti, při shodě vzestupně podle DocA, pak DocB
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Similarity != pairs[j].Similarity {
			return pairs[i].Similarity > pairs[j].Similarity
		}
		if pairs[i].DocA != pairs[j].DocA {
			return pairs[i].DocA < pairs[j].DocA
		}
		return pairs[i].DocB < pairs[j].DocB
	})

	return pairs
}
