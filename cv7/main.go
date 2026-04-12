package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("=== Simulace dat — invertovaný index ===")

	rng := rand.New(rand.NewSource(42))

	fmt.Printf("Generuji %d slov...\n", NumWords)
	words := generateWords(rng)

	fmt.Printf("Sestavuji %d unikátních dvojic (slovo, docID) z %d dokumentů...\n",
		NumPairs, NumDocs)
	start := time.Now()
	inv := BuildInvertedIndex(rng, words)
	fmt.Printf("Hotovo za %v\n", time.Since(start).Round(time.Millisecond))
	fmt.Printf("Počet slov s posting listem: %d\n", len(inv))

	// statistiky posting listů
	totalPostings := 0
	maxLen, minLen := 0, NumPairs
	for _, docs := range inv {
		n := len(docs)
		totalPostings += n
		if n > maxLen {
			maxLen = n
		}
		if n < minLen {
			minLen = n
		}
	}
	fmt.Printf("Celkem postingů: %d  |  Min/Max délka posting listu: %d / %d  |  Průměr: %.1f\n",
		totalPostings, minLen, maxLen, float64(totalPostings)/float64(len(inv)))

	// Kódování invertovaného indexu třemi metodami
	fmt.Println()
	fmt.Println("=== Kódování invertovaného indexu (gap encoding) ===")

	start = time.Now()
	encUnary := EncodeIndex("Unární", inv, UnaryEncodeList)
	durUnary := time.Since(start)

	start = time.Now()
	encGamma := EncodeIndex("EliasGamma", inv, EliasGammaEncodeList)
	durGamma := time.Since(start)

	start = time.Now()
	encFib := EncodeIndex("Fibonacci", inv, FibonacciEncodeList)
	durFib := time.Since(start)

	fmt.Printf("  Unární kódování:      %v\n", durUnary.Round(time.Millisecond))
	fmt.Printf("  Eliasův gamma kód:    %v\n", durGamma.Round(time.Millisecond))
	fmt.Printf("  Fibonacciho kódování: %v\n", durFib.Round(time.Millisecond))

	// ČÁST 3: Srovnání velikostí

	fmt.Println("=== Srovnání velikostí indexů ===")

	rawBytes := RawSize(inv)
	unaryBits := EncodedSize(encUnary)
	gammaBits := EncodedSize(encGamma)
	fibBits := EncodedSize(encFib)
	unaryBytes := EncodedSizeBytes(encUnary)
	gammaBytes := EncodedSizeBytes(encGamma)
	fibBytes := EncodedSizeBytes(encFib)

	fmt.Printf("  %-22s  %10s  %10s  %8s\n", "Metoda", "Bajty (text)", "Bajty (bin)", "vs nezakód.")
	fmt.Println("  ────────────────────────────────────────────────────────")
	fmt.Printf("  %-22s  %10d  %10d  %8s\n", "Nezakódováno (text)", rawBytes, rawBytes, "100%")
	fmt.Printf("  %-22s  %10d  %10d  %7.1f%%\n", "Unární (bity→B)", unaryBits, unaryBytes, 100.0*float64(unaryBytes)/float64(rawBytes))
	fmt.Printf("  %-22s  %10d  %10d  %7.1f%%\n", "Eliasův gamma (B)", gammaBits, gammaBytes, 100.0*float64(gammaBytes)/float64(rawBytes))
	fmt.Printf("  %-22s  %10d  %10d  %7.1f%%\n", "Fibonacci (B)", fibBits, fibBytes, 100.0*float64(fibBytes)/float64(rawBytes))

	fmt.Println()
	fmt.Println("  Poznámka: 'Bajty (text)' = délka bitového řetězce v bajtech (1 bit = 1 znak)")
	fmt.Println("            'Bajty (bin)'  = ekvivalentní binární uložení (ceil(bits/8))")

	// ČÁST 3b: Srovnání rychlosti vyhledávání

	fmt.Println("=== Srovnání rychlosti vyhledávání ===")

	// Vybereme 5 slov a pro každé vyhledáme docID
	searchWords := words[:5]
	const searches = 100

	// připravíme testovací docIDs (existující i neexistující)
	testDocIDs := []int{1, 500, 1000, 5000, 9999}

	fmt.Printf("  Měření: %d opakování vyhledání pro každou kombinaci (slovo × docID)\n\n", searches)
	fmt.Printf("  %-22s  %12s  %12s  %12s  %12s\n", "Metoda", "Celkem [µs]", "Na dotaz [ns]", "Nalezeno", "Nenalezeno")
	fmt.Println("  ────────────────────────────────────────────────────────────────────────────────")

	// Vyhledávání v nezakódovaném indexu
	{
		found, notFound := 0, 0
		t0 := time.Now()
		for rep := 0; rep < searches; rep++ {
			for _, w := range searchWords {
				for _, d := range testDocIDs {
					if SearchRaw(inv, w, d) {
						found++
					} else {
						notFound++
					}
				}
			}
		}
		elapsed := time.Since(t0)
		total := searches * len(searchWords) * len(testDocIDs)
		fmt.Printf("  %-22s  %12d  %12d  %12d  %12d\n",
			"Nezakódováno",
			elapsed.Microseconds(),
			elapsed.Nanoseconds()/int64(total),
			found, notFound)
	}

	// Generická funkce pro měření zakódovaného vyhledávání
	measureEncoded := func(name string, enc EncodedIndex, decodeFn func(string) []int) {
		found, notFound := 0, 0
		t0 := time.Now()
		for rep := 0; rep < searches; rep++ {
			for _, w := range searchWords {
				for _, d := range testDocIDs {
					if SearchEncoded(enc, w, d, decodeFn) {
						found++
					} else {
						notFound++
					}
				}
			}
		}
		elapsed := time.Since(t0)
		total := searches * len(searchWords) * len(testDocIDs)
		fmt.Printf("  %-22s  %12d  %12d  %12d  %12d\n",
			name,
			elapsed.Microseconds(),
			elapsed.Nanoseconds()/int64(total),
			found, notFound)
	}

	measureEncoded("Unární", encUnary, UnaryDecodeList)
	measureEncoded("Eliasův gamma", encGamma, EliasGammaDecodeList)
	measureEncoded("Fibonacci", encFib, FibonacciDecodeList)

	fmt.Println()
	fmt.Println("  Závěr: zakódovaný index vyžaduje dekódování → pomalejší vyhledávání.")
	fmt.Println("  Komprese snižuje velikost, ale zvyšuje čas dotazu (trade-off).")
	fmt.Println("  Eliasův gamma a Fibonacci jsou výrazně úspornější než unár pro velká čísla.")
}
