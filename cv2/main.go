package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	path := "E:\\Downloads\\cs (4).txt\\cs (4).txt"
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
	_, best_uni, uniCount, V := createUnigram(fields)
	bigram_list, best_bi, biCount := createBigram(fields, uniCount, V)
	_, best_tri := createTrigram(fields, biCount, V)
	fmt.Println("Best unigram:", best_uni)
	fmt.Println("Best bigram:", best_bi)
	fmt.Println("Best trigram:", best_tri)

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Zadej text: ")
	input, _ := reader.ReadString('\n')

	fields = strings.Fields(clean(input))
	w1 := fields[len(fields)-1]
	predicted := predictBigram(w1, bigram_list)
	fmt.Println("Predict:", input+predicted)
}

func clean(text string) string {
	text = strings.ToLower(text)
	text = strings.Join(strings.Fields(text), " ")
	return text
}

func createUnigram(fields []string) (map[string]float64, string, map[string]float64, float64) {
	counts := make(map[string]float64)
	for _, w := range fields {
		counts[w]++
	}

	V := float64(len(counts))
	total := float64(len(fields))

	freq := make(map[string]float64)
	bestNgram := ""
	bestFreq := 0.0

	for w, c := range counts {
		freq[w] = (c + 1) / (total + V) // Laplace smoothing
		if freq[w] > bestFreq {
			bestFreq = freq[w]
			bestNgram = w
		}
	}

	return freq, bestNgram, counts, V
}

func createBigram(fields []string, uniCount map[string]float64, V float64) (map[string]map[string]float64, string, map[string]float64) {
	if uniCount == nil || V == 0 {
		uniCount = make(map[string]float64)
		for _, w := range fields {
			uniCount[w]++
		}
		V = float64(len(uniCount))
	}

	bigramCounts := make(map[string]float64)
	bigramFreq := make(map[string]map[string]float64)
	for i := 0; i < len(fields)-1; i++ {
		w1, w2 := fields[i], fields[i+1]
		if bigramFreq[w1] == nil {
			bigramFreq[w1] = make(map[string]float64)
		}
		bigramFreq[w1][w2]++
		bigramCounts[w1+" "+w2]++
	}

	bestNgram := ""
	bestFreq := 0.0

	for w1 := range bigramFreq {
		for w2 := range bigramFreq[w1] {
			bigramFreq[w1][w2] = (bigramFreq[w1][w2] + 1) / (uniCount[w1] + V) // Laplace smoothing
			if bigramFreq[w1][w2] > bestFreq {
				bestFreq = bigramFreq[w1][w2]
				bestNgram = w1 + " " + w2
			}
		}
	}
	return bigramFreq, bestNgram, bigramCounts
}

func predictBigram(w1 string, bigramFreq map[string]map[string]float64) string {
	if bigramFreq[w1] == nil {
		return ""
	}

	bestW2 := ""
	bestFreq := 0.0
	for w2, freq := range bigramFreq[w1] {
		if freq > bestFreq {
			bestFreq = freq
			bestW2 = w2
		}
	}
	return bestW2
}

func createTrigram(fields []string, biCount map[string]float64, V float64) (map[string]map[string]float64, string) {
	if biCount == nil || V == 0 {
		uniCount := make(map[string]float64)
		for _, w := range fields {
			uniCount[w]++
		}
		V = float64(len(uniCount))
		biCount = make(map[string]float64)
		for i := 0; i < len(fields)-1; i++ {
			biCount[fields[i]+" "+fields[i+1]]++
		}
	}

	trigramFreq := make(map[string]map[string]float64)
	for i := 0; i < len(fields)-2; i++ {
		w1, w2, w3 := fields[i], fields[i+1], fields[i+2]
		key := w1 + " " + w2
		if trigramFreq[key] == nil {
			trigramFreq[key] = make(map[string]float64)
		}
		trigramFreq[key][w3]++
	}

	bestNgram := ""
	bestFreq := 0.0

	for w12 := range trigramFreq {
		for w3 := range trigramFreq[w12] {
			trigramFreq[w12][w3] = (trigramFreq[w12][w3] + 1) / (biCount[w12] + V) // Laplace smoothing
			if trigramFreq[w12][w3] > bestFreq {
				bestFreq = trigramFreq[w12][w3]
				bestNgram = w12 + " " + w3
			}
		}
	}
	return trigramFreq, bestNgram
}
