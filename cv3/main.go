package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
)

type SearchResult struct {
	Positions   []int
	Comparisons int
}

func BruteForce(text, pattern string) SearchResult {
	n := len(text)
	m := len(pattern)
	result := SearchResult{}
	if m == 0 || n < m {
		return result
	}
	for i := 0; i <= n-m; i++ {
		j := 0
		for j < m {
			result.Comparisons++
			if text[i+j] != pattern[j] {
				break
			}
			j++
		}
		if j == m {
			result.Positions = append(result.Positions, i)
		}
	}
	return result
}

func computeFailure(pattern string) []int {
	m := len(pattern)
	if m == 0 {
		return nil
	}
	fail := make([]int, m)
	length := 0
	i := 1
	for i < m {
		if pattern[i] == pattern[length] {
			length++
			fail[i] = length
			i++
		} else {
			if length != 0 {
				length = fail[length-1]
			} else {
				fail[i] = 0
				i++
			}
		}
	}
	return fail
}

func KMPSearch(text, pattern string) SearchResult {
	n := len(text)
	m := len(pattern)
	result := SearchResult{}
	if m == 0 || n < m {
		return result
	}
	fail := computeFailure(pattern)
	i, j := 0, 0
	for i < n {
		result.Comparisons++
		if text[i] == pattern[j] {
			i++
			j++
			if j == m {
				result.Positions = append(result.Positions, i-j)
				j = fail[j-1]
			}
		} else {
			if j != 0 {
				j = fail[j-1]
			} else {
				i++
			}
		}
	}
	return result
}

func Horspool(text, pattern string) SearchResult {
	n := len(text)
	m := len(pattern)
	result := SearchResult{}
	if m == 0 || n < m {
		return result
	}

	// Vytvoření shift tabulky (bad character rule)
	shift := make(map[byte]int)
	for i := 0; i < m-1; i++ {
		shift[pattern[i]] = m - 1 - i
	}

	i := m - 1 // pozice v textu zarovnaná s posledním znakem vzoru
	for i < n {
		k := 0
		for k < m {
			result.Comparisons++
			if pattern[m-1-k] != text[i-k] {
				break
			}
			k++
		}
		if k == m {
			// shoda nalezena
			result.Positions = append(result.Positions, i-m+1)
		}
		// posun dle znaku v textu zarovnaného s posledním znakem vzoru
		s, ok := shift[text[i]]
		if !ok {
			s = m // znak není ve vzoru → posun o celou délku vzoru
		}
		i += s
	}
	return result
}

func generateShortText() string {
	base := "Lorem ipsum dolor sit amet, consectetur adipiscing elit, " +
		"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " +
		"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris " +
		"nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in " +
		"reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla " +
		"pariatur. Excepteur sint occaecat cupidatat non proident, sunt in " +
		"culpa qui officia deserunt mollit anim id est laborum. Sed ut " +
		"perspiciatis unde omnis iste natus error sit voluptatem accusantium " +
		"doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo " +
		"inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. "
	var sb strings.Builder
	for sb.Len() < 1000 {
		sb.WriteString(base)
	}
	return sb.String()[:1000]
}

func generateLongText(length int) string {
	words := []string{
		"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
		"and", "cat", "sat", "on", "mat", "in", "park", "with",
		"friend", "happy", "day", "sun", "moon", "star", "light", "dark",
		"tree", "leaf", "wind", "rain", "snow", "fire", "water", "earth",
		"sky", "cloud", "river", "mountain", "valley", "forest", "field", "road",
		"path", "stone", "rock", "sand", "wave", "ocean", "sea", "lake",
	}
	rng := rand.New(rand.NewSource(42))
	var sb strings.Builder
	for sb.Len() < length {
		sb.WriteString(words[rng.Intn(len(words))])
		sb.WriteByte(' ')
	}
	return sb.String()[:length]
}

func generateDNAText(length int) string {
	alphabet := []byte{'A', 'G', 'C', 'T'}
	rng := rand.New(rand.NewSource(42))
	buf := make([]byte, length)
	for i := range buf {
		buf[i] = alphabet[rng.Intn(len(alphabet))]
	}
	return string(buf)
}

type algorithm struct {
	name string
	fn   func(string, string) SearchResult
}

type testCase struct {
	textName string
	text     string
	pattern  string
}

func main() {
	shortText := generateShortText()
	longText := generateLongText(100000)
	dnaText := generateDNAText(100000)

	algorithms := []algorithm{
		{"Brute Force", BruteForce},
		{"KMP", KMPSearch},
		{"BMH (Horspool)", Horspool},
	}

	verifyText := "ABCABCABCABC"
	verifyPattern := "ABCABC"
	bf := BruteForce(verifyText, verifyPattern)
	kmp := KMPSearch(verifyText, verifyPattern)
	bmh := Horspool(verifyText, verifyPattern)

	if len(bf.Positions) == len(kmp.Positions) && len(bf.Positions) == len(bmh.Positions) {
		fmt.Println("Všechny algoritmy nalezly stejný počet výskytů.")
	} else {
		fmt.Println("Algoritmy nalezly různý počet výskytů!")
	}

	tests := []testCase{
		// Krátký text (~1000 znaků)
		{"Krátký (~1000)", shortText, "dolor"},
		{"Krátký (~1000)", shortText, "xyz"},
		{"Krátký (~1000)", shortText, "in"},

		// Dlouhý text (~100 000 znaků)
		{"Dlouhý (~100k)", longText, "mountain"},
		{"Dlouhý (~100k)", longText, "xyzxyz"},
		{"Dlouhý (~100k)", longText, "the"},

		// DNA text (~100 000 znaků, abeceda {A,G,C,T})
		{"DNA (~100k)", dnaText, "AGCT"},
		{"DNA (~100k)", dnaText, "AAAA"},
		{"DNA (~100k)", dnaText, "AGCTAGCTAGCT"},
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)

	// CSV výstup pro vizualizaci
	csvFile, err := os.Create("results.csv")
	if err != nil {
		fmt.Println("Chyba při vytváření CSV:", err)
		return
	}
	defer csvFile.Close()
	csvW := csv.NewWriter(csvFile)
	defer csvW.Flush()
	csvW.Write([]string{"TextType", "TextLen", "Pattern", "PatternLen", "Algorithm", "Comparisons", "Matches"})

	for _, tc := range tests {
		for _, algo := range algorithms {
			res := algo.fn(tc.text, tc.pattern)
			csvW.Write([]string{
				tc.textName,
				strconv.Itoa(len(tc.text)),
				tc.pattern,
				strconv.Itoa(len(tc.pattern)),
				algo.name,
				strconv.Itoa(res.Comparisons),
				strconv.Itoa(len(res.Positions)),
			})
		}
	}
	w.Flush()

	textLengths := []int{1000, 5000, 10000, 25000, 50000, 100000}
	scalingPattern := "AGCT"

	w2 := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)

	scaleCsv, _ := os.Create("scaling.csv")
	defer scaleCsv.Close()
	scaleW := csv.NewWriter(scaleCsv)
	defer scaleW.Flush()
	scaleW.Write([]string{"TextLen", "Algorithm", "Comparisons", "Matches"})

	for _, tLen := range textLengths {
		txt := dnaText[:tLen]
		for _, algo := range algorithms {
			res := algo.fn(txt, scalingPattern)
			scaleW.Write([]string{
				strconv.Itoa(tLen),
				algo.name,
				strconv.Itoa(res.Comparisons),
				strconv.Itoa(len(res.Positions)),
			})
		}
	}
	w2.Flush()

	patterns := []string{
		"AG", "AGCT", "AGCTAGCT", "AGCTAGCTAGCT",
		"AGCTAGCTAGCTAGCTAGCT", "AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
	}

	w3 := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)

	patCsv, _ := os.Create("pattern_scaling.csv")
	defer patCsv.Close()
	patW := csv.NewWriter(patCsv)
	defer patW.Flush()
	patW.Write([]string{"PatternLen", "Pattern", "Algorithm", "Comparisons", "Matches"})

	for _, pat := range patterns {
		for _, algo := range algorithms {
			res := algo.fn(dnaText, pat)
			patW.Write([]string{
				strconv.Itoa(len(pat)),
				pat,
				algo.name,
				strconv.Itoa(res.Comparisons),
				strconv.Itoa(len(res.Positions)),
			})
		}
	}
	w3.Flush()
}
