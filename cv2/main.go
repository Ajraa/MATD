package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
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
	_, best_uni, uniCount, V := createUnigram(fields)
	bigram_list, best_bi, biCount := createBigram(fields, uniCount, V)
	trigram_list, best_tri := createTrigram(fields, biCount, V)
	fmt.Println("Best unigram:", best_uni)
	fmt.Println("Best bigram:", best_bi)
	fmt.Println("Best trigram:", best_tri)

	// Rozdělení dat na trénovací (80%) a testovací (20%)
	splitIdx := int(float64(len(fields)) * 0.8)
	trainFields := fields[:splitIdx]
	testFields := fields[splitIdx:]
	fmt.Printf("\nRozdělení dat: trénovací=%d slov, testovací=%d slov\n", len(trainFields), len(testFields))

	// Vytvoření modelů na trénovacích datech
	uniFreqTrain, _, uniCountTrain, Vtrain := createUnigram(trainFields)
	biFreqTrain, _, biCountTrain := createBigram(trainFields, uniCountTrain, Vtrain)
	triFreqTrain, _ := createTrigram(trainFields, biCountTrain, Vtrain)

	// Výpočet perplexity na testovacích datech
	totalTrain := float64(len(trainFields))
	ppUni := perplexityUnigram(testFields, uniFreqTrain, totalTrain, Vtrain)
	ppBi := perplexityBigram(testFields, biFreqTrain, uniCountTrain, Vtrain)
	ppTri := perplexityTrigram(testFields, triFreqTrain, biCountTrain, Vtrain)

	fmt.Printf("Unigram perplexity: %.2f\n", ppUni)
	fmt.Printf("Bigram  perplexity: %.2f\n", ppBi)
	fmt.Printf("Trigram perplexity: %.2f\n", ppTri)

	// Generátor textu pomocí trigramového modelu
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nZadej počáteční slovo: ")
	input, _ := reader.ReadString('\n')

	predicted := predictBigram(strings.TrimSpace(strings.ToLower(input)), bigram_list)
	fmt.Println("Predikce dalšího slova (bigram):", input+" "+predicted)

	startWord := strings.TrimSpace(strings.ToLower(input))

	generated := generateText(startWord, 3, bigram_list, trigram_list, 50)
	fmt.Println("\nVygenerovaný text:")
	fmt.Println(generated)
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

// weightedRandom - náhodný výběr slova na základě pravděpodobností
func weightedRandom(dist map[string]float64) string {
	total := 0.0
	for _, prob := range dist {
		total += prob
	}

	r := rand.Float64() * total
	cumulative := 0.0
	for word, prob := range dist {
		cumulative += prob
		if r <= cumulative {
			return word
		}
	}

	// Fallback - vrátí první slovo
	for word := range dist {
		return word
	}
	return ""
}

func generateText(startWord string, numSentences int, bigramFreq, trigramFreq map[string]map[string]float64, maxWords int) string {
	words := []string{startWord}
	sentenceCount := 0

	// Druhé slovo získáme z bigram modelu
	if bigramFreq[startWord] != nil {
		w2 := weightedRandom(bigramFreq[startWord])
		words = append(words, w2)
	} else {
		return startWord + " (slovo nenalezeno v modelu)"
	}

	// Generování dalších slov pomocí trigramů
	for sentenceCount < numSentences && len(words) < maxWords {
		w1 := words[len(words)-2]
		w2 := words[len(words)-1]
		key := w1 + " " + w2

		if trigramFreq[key] != nil {
			// Výběr dalšího slova náhodně podle pravděpodobnosti
			w3 := weightedRandom(trigramFreq[key])
			words = append(words, w3)

			// Detekce konce věty
			if strings.HasSuffix(w3, ".") || strings.HasSuffix(w3, "!") || strings.HasSuffix(w3, "?") {
				sentenceCount++
			}
		} else if bigramFreq[w2] != nil {
			// Fallback na bigram model
			w3 := weightedRandom(bigramFreq[w2])
			words = append(words, w3)

			if strings.HasSuffix(w3, ".") || strings.HasSuffix(w3, "!") || strings.HasSuffix(w3, "?") {
				sentenceCount++
			}
		} else {
			// Nelze pokračovat - žádný kontext
			break
		}
	}

	return strings.Join(words, " ")
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

func perplexityUnigram(testFields []string, uniFreq map[string]float64, totalTrain float64, V float64) float64 {
	logSum := 0.0
	N := len(testFields)

	for _, w := range testFields {
		prob, exists := uniFreq[w]
		if !exists {
			// Nepozorované slovo - Laplace smoothing: 1 / (totalTrain + V)
			prob = 1.0 / (totalTrain + V)
		}
		logSum += math.Log2(prob)
	}

	return math.Pow(2, -logSum/float64(N))
}

func perplexityBigram(testFields []string, biFreq map[string]map[string]float64, uniCount map[string]float64, V float64) float64 {
	logSum := 0.0
	N := 0

	for i := 0; i < len(testFields)-1; i++ {
		w1, w2 := testFields[i], testFields[i+1]
		var prob float64

		if biFreq[w1] != nil && biFreq[w1][w2] > 0 {
			// Pozorovaný bigram - již vyhlazená pravděpodobnost
			prob = biFreq[w1][w2]
		} else {
			// Nepozorovaný bigram - Laplace: 1 / (count(w1) + V)
			prob = 1.0 / (uniCount[w1] + V)
		}
		logSum += math.Log2(prob)
		N++
	}

	if N == 0 {
		return math.Inf(1)
	}
	return math.Pow(2, -logSum/float64(N))
}

func perplexityTrigram(testFields []string, triFreq map[string]map[string]float64, biCount map[string]float64, V float64) float64 {
	logSum := 0.0
	N := 0

	for i := 0; i < len(testFields)-2; i++ {
		w1, w2, w3 := testFields[i], testFields[i+1], testFields[i+2]
		key := w1 + " " + w2
		var prob float64

		if triFreq[key] != nil && triFreq[key][w3] > 0 {
			// Pozorovaný trigram - již vyhlazená pravděpodobnost
			prob = triFreq[key][w3]
		} else {
			// Nepozorovaný trigram - Laplace: 1 / (count(w1 w2) + V)
			prob = 1.0 / (biCount[key] + V)
		}
		logSum += math.Log2(prob)
		N++
	}

	if N == 0 {
		return math.Inf(1)
	}
	return math.Pow(2, -logSum/float64(N))
}
