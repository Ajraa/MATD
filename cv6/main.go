package main

import (
	"fmt"
	"sort"
)

// IDF je nevhodné například u lékařských zpráv, kde se určitá slova vyskytují až moc často, např. pacient, léčba, diagnóza atd., takže klíčová slova by měla malé IDF
// Tweety jsou krátké (pokud počítáme se starými Tweety, které byly kratší, než jsou teď), takže TF (term frequency) není příliš informativní, jelikož se slovo vyskytne velmí málo, tím pádem můžem nahratit binární hodnotou
// Taktéž bych výrazně zvýhodnil mentions a hasthagy, jelikož se jedná o anotace od autora, a ty dávají o textu více informací než klasická slova

func main() {

	terms := preprocessCorpus(corpus)

	for i := 0; i < 5 && i < len(corpus); i++ {
		doc := corpus[i]
		docTerms := terms[doc.ID]
		preview := docTerms
		if len(preview) > 12 {
			preview = preview[:12]
		}
		fmt.Printf("Dokument %d – %s:\n", doc.ID, doc.Title)
		fmt.Printf("  Termů celkem: %d  |  Ukázka: %v ...\n\n", len(docTerms), preview)
	}

	//tfidf index
	idx := buildTFIDF(corpus)

	titleOf := make(map[int]string, len(corpus))
	for _, doc := range corpus {
		titleOf[doc.ID] = doc.Title
	}

	// statistiky
	fmt.Println("=== TF-IDF váhy ===")

	selectedTerms := []string{"python", "databáze", "neuronové", "algoritmus", "síť"}

	for _, term := range selectedTerms {
		idf, ok := idx.IDF[term]
		if !ok {
			fmt.Printf("Term %q se v korpusu nevyskytuje.\n\n", term)
			continue
		}
		fmt.Printf("IDF(%q) = %.4f\n", term, idf)

		type entry struct {
			docID int
			tf    float64
			tfidf float64
		}
		var entries []entry
		for docID, tfidfMap := range idx.TFIDF {
			if v, exists := tfidfMap[term]; exists && v > 0 {
				entries = append(entries, entry{docID, idx.TF[docID][term], v})
			}
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].tfidf > entries[j].tfidf
		})
		if len(entries) > 5 {
			entries = entries[:5]
		}
		for rank, e := range entries {
			fmt.Printf("  %d. [ID=%2d] %-35s  TF=%.4f  TF-IDF=%.4f\n",
				rank+1, e.docID, titleOf[e.docID], e.tf, e.tfidf)
		}
		fmt.Println()
	}

	fmt.Println("=== Vyhledávání dotazů ===")

	queries := []string{
		"python",
		"databáze SQL",
		"neuronové sítě strojové učení",
	}

	for _, q := range queries {
		fmt.Printf("Dotaz: %q\n", q)
		results := scoreQuery(q, idx)
		if len(results) == 0 {
			fmt.Println("  Žádné výsledky.\n")
			continue
		}
		top := results
		if len(top) > 5 {
			top = top[:5]
		}
		for rank, r := range top {
			fmt.Printf("  %d. [score=%.4f] [ID=%2d] %s\n",
				rank+1, r.Score, r.DocID, titleOf[r.DocID])
		}
		fmt.Println()
	}

	// kosinová podobnost
	fmt.Println("=== Kosinová podobnost (TF-IDF) – Top 10 párů ===")

	pairs := allPairsSimilarity(idx)
	top10 := pairs
	if len(top10) > 10 {
		top10 = top10[:10]
	}
	for rank, p := range top10 {
		fmt.Printf("  %2d. [sim=%.4f] Doc %2d (%s)  ↔  Doc %2d (%s)\n",
			rank+1, p.Similarity, p.DocA, titleOf[p.DocA], p.DocB, titleOf[p.DocB])
	}
	fmt.Println()

	fmt.Println("=== Nejpodobnější pár (detail) ===")

	if len(pairs) > 0 {
		best := pairs[0]
		fmt.Printf("Doc %d (%s)  ↔  Doc %d (%s)  [sim=%.4f]\n\n",
			best.DocA, titleOf[best.DocA],
			best.DocB, titleOf[best.DocB],
			best.Similarity)

		// sdílené termy – průnik TF-IDF vektorů, seřazený sestupně podle průměrné váhy
		type sharedTerm struct {
			term string
			wA   float64
			wB   float64
			avg  float64
		}
		var shared []sharedTerm
		for term, wA := range idx.TFIDF[best.DocA] {
			if wB, ok := idx.TFIDF[best.DocB][term]; ok && wA > 0 && wB > 0 {
				shared = append(shared, sharedTerm{term, wA, wB, (wA + wB) / 2})
			}
		}
		sort.Slice(shared, func(i, j int) bool {
			return shared[i].avg > shared[j].avg
		})

		fmt.Println("Sdílené termy s nejvyšším TF-IDF (top 10):")
		limit := 10
		if len(shared) < limit {
			limit = len(shared)
		}
		for _, s := range shared[:limit] {
			fmt.Printf("  %-25s  Doc%d=%.4f  Doc%d=%.4f\n",
				s.term, best.DocA, s.wA, best.DocB, s.wB)
		}

	}

	// TF vs TF-IDF
	fmt.Println("=== TF vs TF-IDF porovnání ===")

	pairsTF := allPairsSimilarityTFOnly(idx)

	fmt.Println("Top 5 párů dle čistého TF (bez IDF):")
	for rank, p := range pairsTF[:limitN(len(pairsTF), 5)] {
		fmt.Printf("  %d. [sim=%.4f] Doc %2d (%s)  ↔  Doc %2d (%s)\n",
			rank+1, p.Similarity, p.DocA, titleOf[p.DocA], p.DocB, titleOf[p.DocB])
	}

	fmt.Println("\nTop 5 párů dle TF-IDF:")
	for rank, p := range pairs[:limitN(len(pairs), 5)] {
		fmt.Printf("  %d. [sim=%.4f] Doc %2d (%s)  ↔  Doc %2d (%s)\n",
			rank+1, p.Similarity, p.DocA, titleOf[p.DocA], p.DocB, titleOf[p.DocB])
	}
}

// limitN vrátí minimum z n a max – pomocník pro oříznutí slicí.
func limitN(n, max int) int {
	if n < max {
		return n
	}
	return max
}
