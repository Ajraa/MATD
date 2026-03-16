package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

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

func main() {
	path := "C:\\Users\\ajrac\\Downloads\\cs (1).txt\\cs (1).txt"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	fmt.Println("Using path:", path)

	words, err := loadData(path)
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}

	frequency := createFrequencyMap(words)
	probability := createProbabilityMap(frequency)

	fmt.Printf("Dictionary size (unique words): %d\n", len(frequency))
	fmt.Printf("Token count (all words): %d\n", len(words))

	if len(os.Args) <= 2 {
		fmt.Println("Usage for variants: go run ./cv4 <dataset_path> <word>")
		return
	}

	inputWord := strings.ToLower(os.Args[2])
	alphabet := buildAlphabet(frequency)
	allVariants := editsUpToDistance2(inputWord, alphabet)
	knownVariants := filterKnownVariants(allVariants, frequency)

	fmt.Printf("Input word: %q\n", inputWord)
	fmt.Printf("Generated variants (edit distance <= 2): %d\n", len(allVariants))
	fmt.Printf("Variants found in dictionary: %d\n", len(knownVariants))

	printTopKnownVariants(knownVariants, probability, 10)

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
