package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func findDoc(idx *InvertedIndex, id int) *Document {
	for i := range idx.Documents {
		if idx.Documents[i].ID == id {
			return &idx.Documents[i]
		}
	}
	return nil
}

func firstSentence(text string) string {
	for i, r := range text {
		if r == '.' || r == '!' || r == '?' {
			return text[:i+1]
		}
	}
	if len([]rune(text)) > 80 {
		runes := []rune(text)
		return string(runes[:80]) + "…"
	}
	return text
}

func printResults(ids []int, idx *InvertedIndex) {
	if len(ids) == 0 {
		fmt.Println("  (žádné výsledky)")
		return
	}
	for _, id := range ids {
		doc := findDoc(idx, id)
		if doc == nil {
			continue
		}
		fmt.Printf("  [%2d] %s – %s\n", doc.ID, doc.Title, firstSentence(doc.Content))
	}
}

func printRankedResults(docs []ScoredDoc, idx *InvertedIndex) {
	if len(docs) == 0 {
		fmt.Println("  (žádné výsledky)")
		return
	}
	for rank, sd := range docs {
		doc := findDoc(idx, sd.DocID)
		if doc == nil {
			continue
		}
		fmt.Printf("  %2d. [score=%.4f] [ID=%2d] %s – %s\n",
			rank+1, sd.Score, doc.ID, doc.Title, firstSentence(doc.Content))
	}
}

func interactiveMode(idx *InvertedIndex) {
	fmt.Println("\n=== Interaktivní vyhledávání ===")
	fmt.Println("Zadejte boolean dotaz (AND, OR, NOT, závorky).")
	fmt.Println("Prefix 'r:' pro rankované výsledky (TF-IDF).")
	fmt.Println("Příkaz 'info' zobrazí analýzu indexu. 'konec' ukončí program.\n")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch strings.ToLower(line) {
		case "konec", "exit", "quit":
			fmt.Println("Konec.")
			return
		case "info":
			analyzeIndex(idx)
			continue
		}

		ranked := false
		if strings.HasPrefix(line, "r:") {
			ranked = true
			line = strings.TrimSpace(line[2:])
		}

		if ranked {
			results := evalQueryRanked(line, idx)
			fmt.Printf("Rankované výsledky (%d):\n", len(results))
			printRankedResults(results, idx)
		} else {
			ids, err := evalQuery(line, idx)
			if err != nil {
				fmt.Println("Chyba:", err)
				continue
			}
			fmt.Printf("Boolean výsledky (%d):\n", len(ids))
			printResults(ids, idx)
		}
		fmt.Println()
	}
}

func demo(idx *InvertedIndex) {
	queries := []struct {
		label string
		q     string
	}{
		{"Jednoduché vyhledávání", "python"},
		{"AND – průnik", "python AND data"},
		{"OR – sjednocení", "linux OR windows"},
		{"NOT – doplněk", "NOT kryptografie"},
		{"Závorky + smíšené", "(linux OR android) AND bezpečnost"},
		{"Složený dotaz", "neuronové AND (obraz OR překlad)"},
		{"Rankovaný – TF-IDF", "r:python AND strojové"},
	}

	fmt.Println("\n=== Ukázkové dotazy ===")
	for _, tc := range queries {
		fmt.Printf("\n-- %s: \"%s\"\n", tc.label, tc.q)
		if strings.HasPrefix(tc.q, "r:") {
			q := strings.TrimSpace(tc.q[2:])
			results := evalQueryRanked(q, idx)
			fmt.Printf("Rankované výsledky (%d):\n", len(results))
			printRankedResults(results, idx)
		} else {
			ids, err := evalQuery(tc.q, idx)
			if err != nil {
				fmt.Println("Chyba:", err)
				continue
			}
			fmt.Printf("Boolean výsledky (%d):\n", len(ids))
			printResults(ids, idx)
		}
	}
}

func compareApproaches(idx *InvertedIndex) {
	query := "python"
	fmt.Println("\n=== Porovnání přístupů pro dotaz:", query, "===")

	boolIDs, _ := evalQuery(query, idx)
	fmt.Printf("\nBoolean (neuspořádáno, %d výsledků):\n", len(boolIDs))
	printResults(boolIDs, idx)

	ranked := evalQueryRanked(query, idx)
	fmt.Printf("\nRankované TF-IDF (seřazeno, %d výsledků):\n", len(ranked))
	printRankedResults(ranked, idx)
}

func main() {
	idx := buildIndex(corpus)

	analyzeIndex(idx)
	demo(idx)
	compareApproaches(idx)

	stat, _ := os.Stdin.Stat()
	if (stat.Mode() & os.ModeCharDevice) != 0 {
		interactiveMode(idx)
	}
}
