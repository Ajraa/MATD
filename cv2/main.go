package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	path := "C:\\Users\\ajrac\\Downloads\\cs (1).txt\\cs (1).txt"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	fmt.Println("Using path:", path)

	data, err := os.ReadFile(path)

	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	text := clean(string(data))
	fields := strings.Fields(text)
	_, best_uni := createUnigram(fields)
	_, best_bi := createBigram(fields)
	_, best_tri := createTrigram(fields)
	fmt.Println("Best unigram:", best_uni)
	fmt.Println("Best bigram:", best_bi)
	fmt.Println("Best trigram:", best_tri)

}

func clean(text string) string {
	text = strings.ToLower(text)
	text = strings.Join(strings.Fields(text), " ")
	return text
}

func createUnigram(fields []string) (map[string]float64, string) {
	freq := make(map[string]float64)
	for _, w := range fields {
		freq[w]++
	}

	bestNgram := ""
	bestFreq := 0.0

	total := float64(len(fields))
	for w := range freq {
		freq[w] /= total
		if freq[w] > bestFreq {
			bestFreq = freq[w]
			bestNgram = w
		}
	}

	return freq, bestNgram
}

func createBigram(fields []string) (map[string]map[string]float64, string) {
	bigramFreq := make(map[string]map[string]float64)
	for i := 0; i < len(fields)-1; i++ {
		w1, w2 := fields[i], fields[i+1]
		if bigramFreq[w1] == nil {
			bigramFreq[w1] = make(map[string]float64)
		}
		bigramFreq[w1][w2]++
	}

	total := float64(len(fields))

	bestNgram := ""
	bestFreq := 0.0

	for w1 := range bigramFreq {
		for w2 := range bigramFreq[w1] {
			bigramFreq[w1][w2] /= total
			if bigramFreq[w1][w2] > bestFreq {
				bestFreq = bigramFreq[w1][w2]
				bestNgram = w1 + " " + w2
			}
		}
	}
	return bigramFreq, bestNgram
}

func createTrigram(fields []string) (map[string]map[string]float64, string) {
	trigramFreq := make(map[string]map[string]float64)

	for i := 0; i < len(fields)-2; i++ {
		w1, w2, w3 := fields[i], fields[i+1], fields[i+2]
		if trigramFreq[w1+w2] == nil {
			trigramFreq[w1+w2] = make(map[string]float64)
		}
		trigramFreq[w1+w2][w3]++
	}

	bestNgram := ""
	bestFreq := 0.0

	total := float64(len(fields))
	for w1 := range trigramFreq {
		for w2 := range trigramFreq[w1] {
			trigramFreq[w1][w2] /= total
			if trigramFreq[w1][w2] > bestFreq {
				bestFreq = trigramFreq[w1][w2]
				bestNgram = w1 + " " + w2
			}
		}
	}
	return trigramFreq, bestNgram
}
