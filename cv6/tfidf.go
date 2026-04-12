package main

import (
	"math"
	"sort"
)

type ScoredDoc struct {
	DocID int
	Score float64
}

type TFIDFIndex struct {
	Terms map[int][]string           // DocID → seznam termů (po předzpracování)
	TF    map[int]map[string]float64 // DocID → term → relativní TF
	IDF   map[string]float64         // term → IDF hodnota
	TFIDF map[int]map[string]float64 // DocID → term → TF-IDF váha
	N     int                        // celkový počet dokumentů
}

// TF  = relativní četnost: count(t,d) / total_terms_in_d
// IDF = log(N / df(t))   (přirozený logaritmus; pokud df=0, vrátí 0)
// TF-IDF = TF * IDF
func buildTFIDF(docs []Document) *TFIDFIndex {
	idx := &TFIDFIndex{
		Terms: make(map[int][]string),
		TF:    make(map[int]map[string]float64),
		IDF:   make(map[string]float64),
		TFIDF: make(map[int]map[string]float64),
		N:     len(docs),
	}

	idx.Terms = preprocessCorpus(docs)

	// Výpočet TF (relativní četnost)
	for docID, terms := range idx.Terms {
		total := len(terms)
		if total == 0 {
			idx.TF[docID] = make(map[string]float64)
			continue
		}
		counts := make(map[string]int)
		for _, t := range terms {
			counts[t]++
		}
		tfMap := make(map[string]float64, len(counts))
		for term, cnt := range counts {
			tfMap[term] = float64(cnt) / float64(total)
		}
		idx.TF[docID] = tfMap
	}

	// výpočet DF (document frequency) pro každý term
	df := make(map[string]int)
	for _, tfMap := range idx.TF {
		for term := range tfMap {
			df[term]++
		}
	}

	// výpočet IDF = log(N / df)
	N := float64(idx.N)
	for term, docFreq := range df {
		if docFreq == 0 {
			idx.IDF[term] = 0
		} else {
			idx.IDF[term] = math.Log(N / float64(docFreq))
		}
	}

	// výpočet TF-IDF = TF * IDF
	for docID, tfMap := range idx.TF {
		tfidfMap := make(map[string]float64, len(tfMap))
		for term, tf := range tfMap {
			tfidfMap[term] = tf * idx.IDF[term]
		}
		idx.TFIDF[docID] = tfidfMap
	}

	return idx
}

// scoreQuery ohodnotí dotaz vůči indexu a vrátí dokumenty seřazené sestupně podle skóre.
// score(q,d) = sum_{t in q} tfidf(t,d)
func scoreQuery(query string, idx *TFIDFIndex) []ScoredDoc {
	terms := tokenize(query)

	scores := make(map[int]float64)
	for _, term := range terms {
		for docID, tfidfMap := range idx.TFIDF {
			if val, ok := tfidfMap[term]; ok && val > 0 {
				scores[docID] += val
			}
		}
	}

	result := make([]ScoredDoc, 0, len(scores))
	for docID, score := range scores {
		result = append(result, ScoredDoc{DocID: docID, Score: score})
	}

	// seřazení sestupně podle skóre, při shodě vzestupně podle DocID
	sort.Slice(result, func(i, j int) bool {
		if result[i].Score != result[j].Score {
			return result[i].Score > result[j].Score
		}
		return result[i].DocID < result[j].DocID
	})

	return result
}
