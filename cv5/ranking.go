package main

import (
	"math"
	"sort"
	"strings"
)

// ScoredDoc je dokument s přiřazeným skóre.
type ScoredDoc struct {
	DocID int
	Score float64
}

// tfidf vypočítá TF-IDF skóre pro term v dokumentu.
//
//	tf  = log(1 + freq)
//	idf = log(N / df)
func tfidf(freq, df, N int) float64 {
	if df == 0 {
		return 0
	}
	tf := math.Log1p(float64(freq))
	idf := math.Log(float64(N) / float64(df))
	return tf * idf
}

// evalQueryRanked vyhodnotí dotaz a vrátí dokumenty seřazené podle TF-IDF skóre.
// Termy jsou extrahovány z dotazu a skóre se agreguje součtem přes všechny termy.
func evalQueryRanked(query string, idx *InvertedIndex) []ScoredDoc {
	terms := extractTerms(query)
	N := len(idx.Documents)

	scores := make(map[int]float64)
	for _, term := range terms {
		postings := idx.Postings[strings.ToLower(term)]
		df := len(postings)
		for _, e := range postings {
			scores[e.DocID] += tfidf(e.Frequency, df, N)
		}
	}

	// Kombinujeme s boolean výsledkem – dokumenty mimo boolean výsledek dostanou 0
	boolResult, err := evalQuery(query, idx)
	if err != nil || len(boolResult) == 0 {
		result := make([]ScoredDoc, 0, len(scores))
		for id, score := range scores {
			result = append(result, ScoredDoc{id, score})
		}
		sort.Slice(result, func(i, j int) bool { return result[i].Score > result[j].Score })
		return result
	}

	boolSet := make(map[int]struct{}, len(boolResult))
	for _, id := range boolResult {
		boolSet[id] = struct{}{}
	}

	result := make([]ScoredDoc, 0, len(boolResult))
	for id := range boolSet {
		result = append(result, ScoredDoc{id, scores[id]})
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Score != result[j].Score {
			return result[i].Score > result[j].Score
		}
		return result[i].DocID < result[j].DocID
	})
	return result
}

// extractTerms vytahuje slova, která nejsou operátory ani závorky.
func extractTerms(query string) []string {
	raw := strings.Fields(query)
	var terms []string
	for _, tok := range raw {
		tok = strings.Trim(tok, "()")
		upper := strings.ToUpper(tok)
		if upper != "AND" && upper != "OR" && upper != "NOT" && tok != "" {
			terms = append(terms, tok)
		}
	}
	return terms
}
