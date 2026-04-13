package main

import (
	"math/rand"
	"os"
	"testing"
)

var (
	benchInv    InvertedIndex
	benchUnary  EncodedIndex
	benchGamma  EncodedIndex
	benchFib    EncodedIndex
	benchWords  []string
	benchDocIDs = []int{1, 500, 1000, 5000, 9999}
)

func TestMain(m *testing.M) {
	rng := rand.New(rand.NewSource(42))
	benchWords = generateWords(rng)
	benchInv = BuildInvertedIndex(rng, benchWords)
	benchUnary = EncodeIndex("Unární", benchInv, UnaryEncodeList)
	benchGamma = EncodeIndex("EliasGamma", benchInv, EliasGammaEncodeList)
	benchFib = EncodeIndex("Fibonacci", benchInv, FibonacciEncodeList)
	os.Exit(m.Run())
}

func BenchmarkSearchRaw(b *testing.B) {
	words := benchWords[:5]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			for _, d := range benchDocIDs {
				SearchRaw(benchInv, w, d)
			}
		}
	}
}

func BenchmarkSearchUnary(b *testing.B) {
	words := benchWords[:5]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			for _, d := range benchDocIDs {
				SearchEncoded(benchUnary, w, d, UnaryDecodeList)
			}
		}
	}
}

func BenchmarkSearchEliasGamma(b *testing.B) {
	words := benchWords[:5]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			for _, d := range benchDocIDs {
				SearchEncoded(benchGamma, w, d, EliasGammaDecodeList)
			}
		}
	}
}

func BenchmarkSearchFibonacci(b *testing.B) {
	words := benchWords[:5]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, w := range words {
			for _, d := range benchDocIDs {
				SearchEncoded(benchFib, w, d, FibonacciDecodeList)
			}
		}
	}
}

// TestRychlostiVyhledavani spustí všechna měření jako regulární test.
// Spuštění: go test -v
func TestRychlostiVyhledavani(t *testing.T) {
	pocetSlov := len(benchInv)

	meritka := []struct {
		nazev     string
		fn        func(*testing.B)
		bajtyTotal int
	}{
		{"Nezakódováno", BenchmarkSearchRaw, RawSize(benchInv)},
		{"Unární", BenchmarkSearchUnary, EncodedSizeBytes(benchUnary)},
		{"Eliasův gamma", BenchmarkSearchEliasGamma, EncodedSizeBytes(benchGamma)},
		{"Fibonacci", BenchmarkSearchFibonacci, EncodedSizeBytes(benchFib)},
	}

	t.Logf("%-22s  %15s  %15s  %12s  %12s", "Metoda", "ns/operaci", "alokace/op", "B celkem", "B/slovo")
	t.Logf("%-22s  %15s  %15s  %12s  %12s", "──────────────────────", "───────────────", "───────────────", "────────────", "────────────")
	for _, m := range meritka {
		r := testing.Benchmark(m.fn)
		prumer := m.bajtyTotal / pocetSlov
		t.Logf("%-22s  %15d  %15d  %12d  %12d", m.nazev, r.NsPerOp(), r.AllocsPerOp(), m.bajtyTotal, prumer)
	}
}
