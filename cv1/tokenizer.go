package main

import (
	"sort"
	"strings"
)

type Merge struct {
	A string
	B string
}

type Tokenizer interface {
	Tokenize(text string, k int) ([]string, []string)
}

type WordTokenizer struct{}
type ByteTokenizer struct{}

// llNode je uzel doubly-linked listu pro ByteTokenizer
type llNode struct {
	val  string
	prev *llNode
	next *llNode
	pos  int // pořadí v původní sekvenci (pro greedy left-to-right řazení)
}

func (t WordTokenizer) Tokenize(text string, k int) ([]string, []string) {
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

	// Počáteční frekvence párů (spočítám jednou)
	pairCounts := pairFrequencies(wordSeq, freq)

	merges := make([]Merge, 0, k)
	for i := 0; i < k; i++ {
		if len(pairCounts) == 0 {
			break
		}

		bestPair := Merge{}
		bestCount := 0

		for p, c := range pairCounts {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}
		if bestCount == 0 {
			break
		}

		a, b := bestPair.A, bestPair.B
		merged := a + b
		merges = append(merges, Merge{A: a, B: b})

		updateWordPairCounts(wordSeq, freq, pairCounts, a, b, merged)
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

	// sestavení celé tokenizované sekvence v pořadí původního textu
	var sequence []string
	for _, w := range fields {
		sequence = append(sequence, wordSeq[w]...)
	}

	return vocabList, sequence
}

func (t ByteTokenizer) Tokenize(text string, k int) ([]string, []string) {
	// Inicializace linked listu + invertovaného indexu
	var head *llNode
	var tail *llNode
	nodeIndex := make(map[string]map[*llNode]struct{})

	for i, ch := range text {
		s := string(ch)
		node := &llNode{val: s, pos: i}
		if head == nil {
			head = node
		} else {
			tail.next = node
			node.prev = tail
		}
		tail = node
		if nodeIndex[s] == nil {
			nodeIndex[s] = make(map[*llNode]struct{})
		}
		nodeIndex[s][node] = struct{}{}
	}

	// Počáteční frekvence párů (spočítám jednou)
	pairCounts := make(map[Merge]int)
	for n := head; n != nil && n.next != nil; n = n.next {
		pairCounts[Merge{A: n.val, B: n.next.val}]++
	}

	// K merge operací
	merges := make([]Merge, 0, k)
	for i := 0; i < k; i++ {
		if len(pairCounts) == 0 {
			break
		}

		bestPair := Merge{}
		bestCount := 0
		for p, c := range pairCounts {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}
		if bestCount == 0 {
			break
		}

		a, b := bestPair.A, bestPair.B
		merged := a + b
		merges = append(merges, Merge{A: a, B: b})

		updateBytePairCountsLL(pairCounts, nodeIndex, a, b, merged)
	}

	// Unikátní slovník + sekvence z linked listu
	vocab := make(map[string]struct{})
	var syms []string
	for n := head; n != nil; n = n.next {
		syms = append(syms, n.val)
		vocab[n.val] = struct{}{}
	}

	vocabList := make([]string, 0, len(vocab))
	for s := range vocab {
		vocabList = append(vocabList, s)
	}
	return vocabList, syms
}

func updateWordPairCounts(wordSeq map[string][]string, freq map[string]int, pairCounts map[Merge]int, a, b, merged string) {
	for w, syms := range wordSeq {
		wt := freq[w]
		if wt == 0 || len(syms) < 2 {
			continue
		}

		// Zjistím, zda slovo obsahuje hledaný pár
		hasPair := false
		for j := 0; j+1 < len(syms); j++ {
			if syms[j] == a && syms[j+1] == b {
				hasPair = true
				break
			}
		}
		if !hasPair {
			continue
		}

		// Odečtu staré páry tohoto slova z pairCounts
		for j := 0; j+1 < len(syms); j++ {
			p := Merge{A: syms[j], B: syms[j+1]}
			pairCounts[p] -= wt
			if pairCounts[p] <= 0 {
				delete(pairCounts, p)
			}
		}

		// Aplikuji merge
		newSyms := applyMerge(syms, a, b, merged)
		wordSeq[w] = newSyms

		// Přidám nové páry po merge
		for j := 0; j+1 < len(newSyms); j++ {
			p := Merge{A: newSyms[j], B: newSyms[j+1]}
			pairCounts[p] += wt
		}
	}
}

func updateBytePairCountsLL(pairCounts map[Merge]int, nodeIndex map[string]map[*llNode]struct{}, a, b, merged string) {
	// Sesbírám kandidáty: uzly s hodnotou a, jejichž next má hodnotu b
	candidates := make([]*llNode, 0)
	for n := range nodeIndex[a] {
		if n.next != nil && n.next.val == b {
			candidates = append(candidates, n)
		}
	}
	if len(candidates) == 0 {
		return
	}

	// Seřadím podle pozice pro greedy left-to-right zpracování
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].pos < candidates[j].pos
	})

	// Zpracuji merge s ochranou proti překryvům
	consumed := make(map[*llNode]bool, len(candidates))
	for _, n := range candidates {
		if consumed[n] {
			continue
		}
		// Re-validace (list se mohl změnit předchozím merge)
		if n.next == nil || n.next.val != b || consumed[n.next] {
			continue
		}

		bNode := n.next
		consumed[bNode] = true

		// Odečtu staré páry v okolí
		if n.prev != nil {
			decrementPair(pairCounts, n.prev.val, n.val)
		}
		decrementPair(pairCounts, a, b)
		if bNode.next != nil {
			decrementPair(pairCounts, bNode.val, bNode.next.val)
		}

		// Aktualizuji nodeIndex: odstraním staré záznamy
		delete(nodeIndex[a], n)
		if len(nodeIndex[a]) == 0 {
			delete(nodeIndex, a)
		}
		delete(nodeIndex[b], bNode)
		if bSet := nodeIndex[b]; bSet != nil && len(bSet) == 0 {
			delete(nodeIndex, b)
		}

		// Merge: uzel n přepíšu na merged, odstraním bNode
		n.val = merged
		n.next = bNode.next
		if bNode.next != nil {
			bNode.next.prev = n
		}

		// Přidám n do nodeIndex[merged]
		if nodeIndex[merged] == nil {
			nodeIndex[merged] = make(map[*llNode]struct{})
		}
		nodeIndex[merged][n] = struct{}{}

		// Přidám nové páry
		if n.prev != nil {
			pairCounts[Merge{A: n.prev.val, B: n.val}]++
		}
		if n.next != nil {
			pairCounts[Merge{A: n.val, B: n.next.val}]++
		}
	}
}

func decrementPair(pairCounts map[Merge]int, a, b string) {
	p := Merge{A: a, B: b}
	pairCounts[p]--
	if pairCounts[p] <= 0 {
		delete(pairCounts, p)
	}
}

func pairFrequencies(wordSeq map[string][]string, freq map[string]int) map[Merge]int {
	pairCounts := make(map[Merge]int)
	for w, syms := range wordSeq {
		wt := freq[w]
		if wt == 0 || len(syms) < 2 {
			continue
		}
		for i := 0; i+1 < len(syms); i++ {
			p := Merge{A: syms[i], B: syms[i+1]}
			pairCounts[p] += wt
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
