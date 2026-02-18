package main

import "strings"

type Merge struct {
	A string
	B string
}

type Tokenizer interface {
	Tokenize(text string, k int) []string
}

type WordTokenizer struct{}
type ByteTokenizer struct{}

func (t WordTokenizer) Tokenize(text string, k int) []string {
	fields := strings.Fields(text)

	freq := make(map[string]int)
	for _, w := range fields {
		freq[w]++
	}

	// vytvořím mapu pro uložení sekvence symbolů pro každé slovo
	wordSeq := make(map[string][]string, len(freq))
	for w := range freq {
		r := []rune(w)
		syms := make([]string, 0, len(r)+1)
		for _, rr := range r {
			syms = append(syms, string(rr))
		}
		syms = append(syms, "<end_of_word>")
		wordSeq[w] = syms
	}

	merges := make([]Merge, 0, k)
	for i := 0; i < k; i++ {
		pairCounts := pairFrequencies(wordSeq, freq)
		if len(pairCounts) == 0 {
			break
		}

		bestPair := ""
		bestCount := 0

		for p, c := range pairCounts {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}

		parts := strings.SplitN(bestPair, " ", 2)
		a, b := parts[0], parts[1]
		merges = append(merges, Merge{A: a, B: b})

		for w, syms := range wordSeq {
			wordSeq[w] = applyMerge(syms, a, b, a+b)
		}
	}

	vocab := make(map[string]struct{})
	for _, syms := range wordSeq {
		for _, s := range syms {
			vocab[s] = struct{}{} // empty struct{} is used to save memory, as it occupies zero bytes, map takes only uniques
		}
	}

	vocabList := make([]string, 0, len(vocab))
	for s := range vocab {
		vocabList = append(vocabList, s)
	}

	return vocabList
}

func pairFrequencies(wordSeq map[string][]string, freq map[string]int) map[string]int {
	pairCounts := make(map[string]int)
	for w, syms := range wordSeq {
		wt := freq[w]
		if wt == 0 || len(syms) < 2 {
			continue
		}
		local := make(map[string]int)
		for i := 0; i+1 < len(syms); i++ {
			p := syms[i] + " " + syms[i+1]
			local[p]++
		}
		for p, c := range local {
			pairCounts[p] += c * wt
		}
	}
	return pairCounts
}

func applyMerge(syms []string, a, b, merged string) []string {
	result := make([]string, 0, len(syms))
	i := 0
	for i < len(syms) {
		if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
			result = append(result, merged)
			i += 2
		} else {
			result = append(result, syms[i])
			i++
		}
	}
	return result
}

func (t ByteTokenizer) Tokenize(text string, k int) []string {
	freq := make(map[string]int)
	for ch := range text {
		freq[string(ch)]++
	}

	merges := make([]Merge, 0, k)
	for i := 0; i < k; i++ {
		pairCounts := pairFrequencies()
	}
}
