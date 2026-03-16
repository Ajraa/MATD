package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"
)

const defaultDatasetPath = "C:\\Users\\ajrac\\Downloads\\cs (1).txt\\cs (1).txt"

// diakritika jsou 2 znaky, proto pracujeme s runami, ne bajty
func levenshteinDistance(s1, s2 string) int {
	r1 := []rune(s1)
	r2 := []rune(s2)

	return levenshteinDistanceRunes(r1, r2)
}

func levenshteinDistanceRunes(r1, r2 []rune) int {
	if len(r1) == 0 {
		return len(r2)
	}
	if len(r2) == 0 {
		return len(r1)
	}

	if r1[len(r1)-1] == r2[len(r2)-1] {
		return levenshteinDistanceRunes(r1[:len(r1)-1], r2[:len(r2)-1])
	}

	return 1 + min(
		levenshteinDistanceRunes(r1[:len(r1)-1], r2),             // Smazání
		levenshteinDistanceRunes(r1, r2[:len(r2)-1]),             // Vložení
		levenshteinDistanceRunes(r1[:len(r1)-1], r2[:len(r2)-1]), // Nahrazení
	)
}

// Iterativní varianta Levenshteina pro rychlé porovnání se slovníkem.
func levenshteinDistanceDP(s1, s2 string) int {
	r1 := []rune(s1)
	r2 := []rune(s2)

	if len(r1) == 0 {
		return len(r2)
	}
	if len(r2) == 0 {
		return len(r1)
	}

	prev := make([]int, len(r2)+1)
	for j := 0; j <= len(r2); j++ {
		prev[j] = j
	}

	for i := 1; i <= len(r1); i++ {
		curr := make([]int, len(r2)+1)
		curr[0] = i

		for j := 1; j <= len(r2); j++ {
			cost := 0
			if r1[i-1] != r2[j-1] {
				cost = 1
			}

			curr[j] = min(
				prev[j]+1,
				curr[j-1]+1,
				prev[j-1]+cost,
			)
		}

		prev = curr
	}

	return prev[len(r2)]
}

func bestCandidateFromVariants(inputWord string, frequency map[string]int, alphabet []rune) (string, bool) {
	if _, exists := frequency[inputWord]; exists {
		return inputWord, true
	}

	variants := editsUpToDistance2(inputWord, alphabet)
	known := filterKnownVariants(variants, frequency)
	if len(known) == 0 {
		return "", false
	}

	sort.Slice(known, func(i, j int) bool {
		fi := frequency[known[i]]
		fj := frequency[known[j]]
		if fi == fj {
			return known[i] < known[j]
		}
		return fi > fj
	})

	return known[0], true
}

func bestCandidateByDictionaryDistance(inputWord string, frequency map[string]int) (string, int, bool) {
	if len(frequency) == 0 {
		return "", 0, false
	}

	bestWord := ""
	bestDistance := math.MaxInt
	bestFrequency := -1

	for candidate, freq := range frequency {
		d := levenshteinDistanceDP(inputWord, candidate)
		if d < bestDistance ||
			(d == bestDistance && freq > bestFrequency) ||
			(d == bestDistance && freq == bestFrequency && candidate < bestWord) {
			bestWord = candidate
			bestDistance = d
			bestFrequency = freq
		}
	}

	return bestWord, bestDistance, true
}

func correctSentenceByVariants(sentence string, frequency map[string]int) string {
	alphabet := buildAlphabet(frequency)
	return correctSentence(sentence, func(word string) string {
		candidate, ok := bestCandidateFromVariants(word, frequency, alphabet)
		if !ok {
			return word
		}
		return candidate
	})
}

func correctSentenceByDictionaryDistance(sentence string, frequency map[string]int) string {
	return correctSentence(sentence, func(word string) string {
		candidate, _, ok := bestCandidateByDictionaryDistance(word, frequency)
		if !ok {
			return word
		}
		return candidate
	})
}

func correctSentence(sentence string, correctWord func(string) string) string {
	runes := []rune(sentence)
	var b strings.Builder

	for i := 0; i < len(runes); {
		if !unicode.IsLetter(runes[i]) {
			b.WriteRune(runes[i])
			i++
			continue
		}

		start := i
		for i < len(runes) && unicode.IsLetter(runes[i]) {
			i++
		}

		original := string(runes[start:i])
		lower := strings.ToLower(original)
		corrected := correctWord(lower)

		if len(corrected) > 0 && unicode.IsUpper(runes[start]) {
			correctedRunes := []rune(corrected)
			correctedRunes[0] = unicode.ToUpper(correctedRunes[0])
			corrected = string(correctedRunes)
		}

		b.WriteString(corrected)
	}

	return b.String()
}

// Pro n >= 1 platí uzavřený tvar: 2*a*n + n + a - 1.
func countGeneratedVariantsDistance1(wordLen, alphabetSize int) int {
	if wordLen < 0 || alphabetSize < 0 {
		return 0
	}
	if wordLen == 0 {
		return alphabetSize
	}

	deletions := wordLen
	transpositions := wordLen - 1
	replacements := wordLen * (alphabetSize - 1)
	insertions := (wordLen + 1) * alphabetSize

	return deletions + transpositions + replacements + insertions
}

// Horní odhad pro počet kandidátů do vzdálenosti 2 bez zohlednění deduplikace.
func countGeneratedVariantsDistance2UpperBound(wordLen, alphabetSize int) int {
	if wordLen < 0 || alphabetSize < 0 {
		return 0
	}

	c1 := countGeneratedVariantsDistance1(wordLen, alphabetSize)
	maxSecondStep := countGeneratedVariantsDistance1(wordLen+1, alphabetSize)
	return c1 + c1*maxSecondStep
}

// trainFrequencyModel načte korpus a vytvoří frekvenční slovník.
func trainFrequencyModel(path string) (map[string]int, error) {
	words, err := loadData(path)
	if err != nil {
		return nil, err
	}

	return createFrequencyMap(words), nil
}

func main() {
	path := defaultDatasetPath
	if len(os.Args) >= 2 {
		path = os.Args[1]
	}
	fmt.Println("Using path:", path)

	frequency, err := trainFrequencyModel(path)
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}

	tokenCount := 0
	for _, count := range frequency {
		tokenCount += count
	}
	//probability := createProbabilityMap(frequency)

	fmt.Printf("Dictionary size (unique words): %d\n", len(frequency))
	fmt.Printf("Token count (all words): %d\n", tokenCount)

	sentence := "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí."
	if len(os.Args) >= 3 {
		sentence = strings.Join(os.Args[2:], " ")
	}

	byVariants := correctSentenceByVariants(sentence, frequency)
	byDictionaryDistance := correctSentenceByDictionaryDistance(sentence, frequency)

	fmt.Println("Original:")
	fmt.Println(sentence)

	// pět je v datasetu, proto se neopraví, možná by mohlo pomoct neopravování slov, co jsou v datasetu
	fmt.Println("Varianty:")
	fmt.Println(byVariants)

	// ke slovo televezí se opraví na televizí, protože to je o 1 blíže než televizi, pomohli by n-gramy
	fmt.Println("Vzdálenost:")
	fmt.Println(byDictionaryDistance)

	alphabetSize := len(buildAlphabet(frequency))
	fmt.Printf("Estimated generated variants for one word (distance 1): %d\n", countGeneratedVariantsDistance1(6, alphabetSize))
	fmt.Printf("Estimated generated variants for one word (distance <= 2, upper bound): %d\n", countGeneratedVariantsDistance2UpperBound(6, alphabetSize))
}

func loadData(filepath string) ([]string, error) {
	data, err := os.ReadFile(filepath)

	if err != nil {
		fmt.Println("Error reading file:", err)
		return nil, err
	}

	cleanData := clean(string(data))
	return tokenizeWords(cleanData), nil
}

func clean(text string) string {
	text = strings.ToLower(text)
	text = strings.Join(strings.Fields(text), " ")
	return text
}

func tokenizeWords(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r)
	})
}

func createFrequencyMap(words []string) map[string]int {
	frequency := make(map[string]int)
	for _, word := range words {
		if word == "" {
			continue
		}
		frequency[word]++
	}

	return frequency
}

func createProbabilityMap(frequency map[string]int) map[string]float64 {
	total := 0
	for _, count := range frequency {
		total += count
	}

	probability := make(map[string]float64, len(frequency))
	if total == 0 {
		return probability
	}

	for word, count := range frequency {
		probability[word] = float64(count) / float64(total)
	}

	return probability
}

func buildAlphabet(frequency map[string]int) []rune {
	charset := make(map[rune]struct{})
	for word := range frequency {
		for _, r := range word {
			charset[r] = struct{}{}
		}
	}

	alphabet := make([]rune, 0, len(charset))
	for r := range charset {
		alphabet = append(alphabet, r)
	}
	sort.Slice(alphabet, func(i, j int) bool { return alphabet[i] < alphabet[j] })
	return alphabet
}

func editsUpToDistance2(word string, alphabet []rune) map[string]struct{} {
	variants := make(map[string]struct{})
	oneEdit := edits1(word, alphabet)
	for candidate := range oneEdit {
		variants[candidate] = struct{}{}
	}

	for candidate := range oneEdit {
		twoEdits := edits1(candidate, alphabet)
		for candidate2 := range twoEdits {
			variants[candidate2] = struct{}{}
		}
	}

	delete(variants, word)
	return variants
}

func edits1(word string, alphabet []rune) map[string]struct{} {
	runes := []rune(word)
	result := make(map[string]struct{})

	for i := 0; i < len(runes); i++ {
		candidate := string(append(append([]rune{}, runes[:i]...), runes[i+1:]...))
		result[candidate] = struct{}{}
	}

	for i := 0; i < len(runes)-1; i++ {
		candidateRunes := append([]rune{}, runes...)
		candidateRunes[i], candidateRunes[i+1] = candidateRunes[i+1], candidateRunes[i]
		result[string(candidateRunes)] = struct{}{}
	}

	for i := 0; i < len(runes); i++ {
		for _, ch := range alphabet {
			if runes[i] == ch {
				continue
			}
			candidateRunes := append([]rune{}, runes...)
			candidateRunes[i] = ch
			result[string(candidateRunes)] = struct{}{}
		}
	}

	for i := 0; i <= len(runes); i++ {
		for _, ch := range alphabet {
			candidateRunes := make([]rune, 0, len(runes)+1)
			candidateRunes = append(candidateRunes, runes[:i]...)
			candidateRunes = append(candidateRunes, ch)
			candidateRunes = append(candidateRunes, runes[i:]...)
			result[string(candidateRunes)] = struct{}{}
		}
	}

	delete(result, word)
	return result
}

func filterKnownVariants(variants map[string]struct{}, frequency map[string]int) []string {
	known := make([]string, 0)
	for variant := range variants {
		if _, exists := frequency[variant]; exists {
			known = append(known, variant)
		}
	}
	return known
}

func printTopKnownVariants(known []string, probability map[string]float64, limit int) {
	if len(known) == 0 {
		return
	}

	sort.Slice(known, func(i, j int) bool {
		pi := probability[known[i]]
		pj := probability[known[j]]
		if pi == pj {
			return known[i] < known[j]
		}
		return pi > pj
	})

	if limit > len(known) {
		limit = len(known)
	}

	fmt.Printf("Top %d known variants by probability:\n", limit)
	for i := 0; i < limit; i++ {
		word := known[i]
		fmt.Printf("%2d. %s (p=%.8f)\n", i+1, word, probability[word])
	}
}
