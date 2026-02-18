package main

import (
	"fmt"
	"sort"
	"strings"
)

type pair struct {
	Word  string
	Count int
}

func printStatistics(text string) {
	tokens := strings.Fields(text)

	freq := make(map[string]int)
	for _, w := range tokens {
		freq[w]++
	}

	total := len(tokens)
	unique := len(freq)

	pairs := make([]pair, 0, len(freq))
	for w, c := range freq {
		pairs = append(pairs, pair{w, c})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Count == pairs[j].Count {
			return pairs[i].Word < pairs[j].Word
		}
		return pairs[i].Count > pairs[j].Count
	})

	limit := 20
	if limit > len(pairs) {
		limit = len(pairs)
	}

	fmt.Println("Počet slov:", total)
	fmt.Println("Počet unikátních slov:", unique)
	fmt.Println("20 nejčetnějších slov:")
	for i := 0; i < limit; i++ {
		fmt.Printf("%2d. %q — %d\n", i+1, pairs[i].Word, pairs[i].Count)
	}
}
