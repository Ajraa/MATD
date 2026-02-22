package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

// Výchozí cesta k českému datasetu (stejná jako v launch.json).
// Lze přepsat proměnnou prostředí DATASET_PATH.
const defaultDatasetPath = `E:\Downloads\cs (4).txt\cs (4).txt`

// Fallback text pro případ, že soubor neexistuje
const fallbackText = `the cat sat on the mat the cat ate the rat and the bat sat on the flat hat ` +
	`the cat sat on the mat again and the rat ran from the bat the cat and the rat sat together ` +
	`on the mat while the bat flew over the flat hat the cat chased the rat around the mat and ` +
	`the bat watched from the hat`

const mergeOps = 1000

// ---------- Načtení a čištění datasetu ----------

var (
	datasetOnce sync.Once
	datasetText string

	tokenizeOnce    sync.Once
	cachedWordVocab []string
	cachedWordSeq   []string
	cachedByteVocab []string
	cachedByteSeq   []string
	cachedText      string
)

// loadDataset načte a vyčistí český dataset (jednou pro všechny testy).
func loadDataset(t *testing.T) string {
	t.Helper()
	datasetOnce.Do(func() {
		path := os.Getenv("DATASET_PATH")
		if path == "" {
			path = defaultDatasetPath
		}

		data, err := os.ReadFile(path)
		if err != nil {
			datasetText = fallbackText
			return
		}

		// Základní čištění: lowercase + odstranění vícenásobných mezer
		datasetText = clean(string(data))
	})
	return datasetText
}

// tokenizeResult spustí oba tokenizery paralelně (jednou pro všechny testy)
// a výsledky uloží do cache.
type tokenizeResult struct {
	WordVocab, WordSeq []string
	ByteVocab, ByteSeq []string
	Text               string
}

func loadTokenized(t *testing.T) tokenizeResult {
	t.Helper()
	tokenizeOnce.Do(func() {
		cachedText = loadDataset(t)
		//cachedText = truncateText(fullText, 5000)

		cachedWordVocab, cachedWordSeq = WordTokenizer{}.Tokenize(cachedText, mergeOps)
		cachedByteVocab, cachedByteSeq = ByteTokenizer{}.Tokenize(cachedText, mergeOps)
	})
	return tokenizeResult{
		WordVocab: cachedWordVocab,
		WordSeq:   cachedWordSeq,
		ByteVocab: cachedByteVocab,
		ByteSeq:   cachedByteSeq,
		Text:      cachedText,
	}
}

func TestDatasetNacteni(t *testing.T) {
	text := loadDataset(t)

	numWords := len(strings.Fields(text))
	numChars := utf8.RuneCountInString(text)
	numBytes := len(text)

	t.Logf("=== Dataset ===")
	t.Logf("Velikost: %d bytů, %d znaků, %d slov", numBytes, numChars, numWords)

	// Ukázka prvních 200 znaků
	preview := text
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	t.Logf("Ukázka: %s", preview)

	if numWords < 1_000_000 {
		t.Logf("VAROVÁNÍ: dataset má pouze %d slov (minimum 1 000 000)", numWords)
	} else {
		t.Logf("OK: dataset má %d slov (>= 1 000 000)", numWords)
	}
}

// ---------- Tokenizační efektivita ----------

func TestTokenizacniEfektivita(t *testing.T) {
	r := loadTokenized(t)
	text := r.Text
	wordVocab, wordSeq := r.WordVocab, r.WordSeq
	byteVocab, byteSeq := r.ByteVocab, r.ByteSeq

	numChars := utf8.RuneCountInString(text)
	numBytes := len(text)
	numWords := len(strings.Fields(text))

	// Počet tokenů na 1000 znaků
	wordTokensPer1000 := float64(len(wordSeq)) / float64(numChars) * 1000
	byteTokensPer1000 := float64(len(byteSeq)) / float64(numChars) * 1000

	// Počet tokenů na slovo = (#tokenů v tokenizovaném textu) / (#slov v původním textu)
	wordTokensPerWord := float64(len(wordSeq)) / float64(numWords)
	byteTokensPerWord := float64(len(byteSeq)) / float64(numWords)

	t.Logf("=== Tokenizační efektivita (K=%d) ===", mergeOps)
	t.Logf("Délka textu: %d znaků, %d bytů, %d slov", numChars, numBytes, numWords)
	t.Logf("")
	t.Logf("%-25s %15s %15s", "", "WordBPE", "ByteBPE")
	t.Logf("%-25s %15d %15d", "Velikost slovníku", len(wordVocab), len(byteVocab))
	t.Logf("%-25s %15d %15d", "Počet tokenů v sekvenci", len(wordSeq), len(byteSeq))
	t.Logf("%-25s %15.2f %15.2f", "Tokenů na 1000 znaků", wordTokensPer1000, byteTokensPer1000)
	t.Logf("%-25s %15.2f %15.2f", "Tokenů na slovo", wordTokensPerWord, byteTokensPerWord)

	// Základní sanity checky
	if len(wordSeq) == 0 {
		t.Error("WordTokenizer vrátil prázdnou sekvenci")
	}
	if len(byteSeq) == 0 {
		t.Error("ByteTokenizer vrátil prázdnou sekvenci")
	}
}

// ---------- Mezislovní tokeny ----------

func TestMezislovniTokeny(t *testing.T) {
	r := loadTokenized(t)
	wordSeq := r.WordSeq
	byteSeq := r.ByteSeq

	wordSpaceCount := 0
	for _, tok := range wordSeq {
		if strings.Contains(tok, " ") {
			wordSpaceCount++
		}
	}

	byteSpaceCount := 0
	for _, tok := range byteSeq {
		if strings.Contains(tok, " ") {
			byteSpaceCount++
		}
	}

	wordSpaceRatio := float64(wordSpaceCount) / float64(len(wordSeq)) * 100
	byteSpaceRatio := float64(byteSpaceCount) / float64(len(byteSeq)) * 100

	t.Logf("=== Mezislovní tokeny (K=%d) ===", mergeOps)
	t.Logf("%-35s %15s %15s", "", "WordBPE", "ByteBPE")
	t.Logf("%-35s %15d %15d", "Tokenů obsahujících mezeru", wordSpaceCount, byteSpaceCount)
	t.Logf("%-35s %14.2f%% %14.2f%%", "Podíl tokenů s mezerou", wordSpaceRatio, byteSpaceRatio)

	// WordBPE by nikdy neměl mít tokeny s mezerou (tokenizuje po slovech)
	if wordSpaceCount != 0 {
		t.Errorf("WordTokenizer: očekáváno 0 tokenů s mezerou, ale nalezeno %d", wordSpaceCount)
	}

	// ByteBPE může (a typicky bude) mít tokeny s mezerou
	t.Logf("ByteBPE má %d tokenů obsahujících mezeru — to je očekávané chování", byteSpaceCount)
}

// ---------- Kvalitativní srovnání: segmentace vybraných slov ----------

func TestKvalitativniSrovnani(t *testing.T) {
	r := loadTokenized(t)
	text := r.Text
	wordSeq := r.WordSeq
	byteSeq := r.ByteSeq

	// Vybraná česká slova k analýze
	selectedWords := []string{"přípravek", "použití", "léčivý", "registrace", "evropské", "může"}

	t.Logf("=== Kvalitativní srovnání segmentace (K=%d) ===", mergeOps)

	// --- WordBPE segmentace ---
	t.Logf("")
	t.Logf("--- WordBPE segmentace ---")

	fields := strings.Fields(text)
	wordSegMap := buildWordSegMap(wordSeq, fields)

	for _, w := range selectedWords {
		seg, ok := wordSegMap[w]
		if ok {
			t.Logf("  %-12s → [%s]", w, strings.Join(seg, " | "))
		} else {
			t.Logf("  %-12s → (nenalezeno)", w)
		}
	}

	// --- ByteBPE segmentace ---
	// Najdeme, jak ByteBPE tokenizuje vybraná slova v kontextu celého textu.
	// Protože ByteBPE pracuje s celým textem včetně mezer, extrahujeme tokeny
	// které pokrývají oblast daného slova.
	t.Logf("")
	t.Logf("--- ByteBPE segmentace ---")

	for _, w := range selectedWords {
		seg := extractByteSegmentation(byteSeq, w)
		if len(seg) > 0 {
			t.Logf("  %-12s → [%s]", w, strings.Join(seg, " | "))
		} else {
			t.Logf("  %-12s → (nenalezeno)", w)
		}
	}
}

// ---------- Vliv K na velikost slovníku ----------

func TestVlivKNaEfektivitu(t *testing.T) {
	fullText := loadDataset(t)
	text := truncateText(fullText, 5000)

	kValues := []int{10, 25, 50, 100}

	t.Logf("=== Vliv K na tokenizaci ===")
	t.Logf("%-6s %15s %15s %15s %15s", "K", "Word vocab", "Word tokenů", "Byte vocab", "Byte tokenů")

	for _, k := range kValues {
		wV, wS := WordTokenizer{}.Tokenize(text, k)
		bV, bS := ByteTokenizer{}.Tokenize(text, k)
		t.Logf("%-6d %15d %15d %15d %15d", k, len(wV), len(wS), len(bV), len(bS))
	}
}

// truncateText ořízne text na prvních maxChars znaků (rune-safe),
// aby testy na velkém datasetu neběžely příliš dlouho.
func truncateText(text string, maxChars int) string {
	runes := []rune(text)
	if len(runes) <= maxChars {
		return text
	}
	return string(runes[:maxChars])
}

// ============ Pomocné funkce pro testy ============

// buildWordSegMap mapuje každé slovo z fields na jeho tokeny z wordSeq.
// WordBPE může sloučit <end_of_word> do posledního tokenu (např. "the<end_of_word>"),
// takže hledáme konec slova podle suffixu "<end_of_word>".
func buildWordSegMap(seq []string, fields []string) map[string][]string {
	result := make(map[string][]string)
	pos := 0
	for _, w := range fields {
		if _, seen := result[w]; seen {
			// přeskočíme tokeny tohoto slova
			for pos < len(seq) {
				tok := seq[pos]
				pos++
				if strings.HasSuffix(tok, "<end_of_word>") {
					break
				}
			}
			continue
		}
		var group []string
		for pos < len(seq) {
			tok := seq[pos]
			pos++
			group = append(group, tok)
			if strings.HasSuffix(tok, "<end_of_word>") {
				break
			}
		}
		result[w] = group
	}
	return result
}

// extractByteSegmentation najde první izolovaný výskyt slova (ohraničený
// mezerou nebo okrajem textu) v rekonstruovaném textu z byteSeq a vrátí
// tokeny, které pokrývají přesně toto slovo (bez okolních mezer).
func extractByteSegmentation(seq []string, word string) []string {
	fullText := strings.Join(seq, "")

	// Hledáme slovo jako celé slovo (ne podřetězec jiného slova)
	searchFrom := 0
	idx := -1
	for {
		i := strings.Index(fullText[searchFrom:], word)
		if i == -1 {
			break
		}
		pos := searchFrom + i
		// Kontrola hranic: před slovem musí být začátek nebo mezera,
		// za slovem konec nebo mezera
		leftOK := pos == 0 || fullText[pos-1] == ' '
		rightOK := pos+len(word) == len(fullText) || fullText[pos+len(word)] == ' '
		if leftOK && rightOK {
			idx = pos
			break
		}
		searchFrom = pos + 1
	}
	if idx == -1 {
		return nil
	}

	// Najdeme tokeny pokrývající přesně rozsah [idx, idx+len(word))
	bytePos := 0
	var result []string
	for _, tok := range seq {
		tokEnd := bytePos + len(tok)
		if tokEnd > idx && bytePos < idx+len(word) {
			// Ořízneme token na průnik s rozsahem slova
			start := 0
			if bytePos < idx {
				start = idx - bytePos
			}
			end := len(tok)
			if tokEnd > idx+len(word) {
				end = idx + len(word) - bytePos
			}
			result = append(result, fmt.Sprintf("%q", tok[start:end]))
		}
		bytePos = tokEnd
		if bytePos >= idx+len(word) {
			break
		}
	}
	return result
}
