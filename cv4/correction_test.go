package main

import (
	"sync"
	"testing"
)

const benchmarkSentence = "Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí."

var (
	realDataOnce sync.Once
	realDataFreq map[string]int
	realDataErr  error
)

func loadRealFrequencyModel(tb testing.TB) map[string]int {
	tb.Helper()

	realDataOnce.Do(func() {
		realDataFreq, realDataErr = trainFrequencyModel(defaultDatasetPath)
	})

	if realDataErr != nil {
		tb.Fatalf("načtení reálného datasetu selhalo z %q: %v", defaultDatasetPath, realDataErr)
	}

	if len(realDataFreq) == 0 {
		tb.Fatalf("model z reálného datasetu je prázdný")
	}

	return realDataFreq
}

func TestTrainFrequencyModelFromRealDataset(t *testing.T) {
	frequency := loadRealFrequencyModel(t)

	if len(frequency) < 100 {
		t.Fatalf("očekáván větší reálný slovník, získáno %d položek", len(frequency))
	}
}

func TestCorrectSentenceApproachesWithRealDataset(t *testing.T) {
	frequency := loadRealFrequencyModel(t)

	variantResult := correctSentenceByVariants(benchmarkSentence, frequency)
	distanceResult := correctSentenceByDictionaryDistance(benchmarkSentence, frequency)

	if variantResult == "" || distanceResult == "" {
		t.Fatalf("oprava věty vrátila prázdný výsledek")
	}

	if variantResult == benchmarkSentence && distanceResult == benchmarkSentence {
		t.Fatalf("ani jeden přístup neprovedl žádnou opravu")
	}
}

func TestCountGeneratedVariantsDistance1(t *testing.T) {
	const n = 5
	const a = 26
	got := countGeneratedVariantsDistance1(n, a)
	want := 2*a*n + n + a - 1

	if got != want {
		t.Fatalf("počet variant pro n=%d, a=%d: očekáváno %d, získáno %d", n, a, want, got)
	}
}

func TestCountGeneratedVariantsDistance2UpperBound(t *testing.T) {
	const n = 5
	const a = 26
	c1 := countGeneratedVariantsDistance1(n, a)
	want := c1 + c1*countGeneratedVariantsDistance1(n+1, a)
	got := countGeneratedVariantsDistance2UpperBound(n, a)

	if got != want {
		t.Fatalf("horní odhad počtu variant do vzdálenosti 2: očekáváno %d, získáno %d", want, got)
	}
}

func TestTrainFrequencyModelFromPath(t *testing.T) {
	frequency, err := trainFrequencyModel(defaultDatasetPath)
	if err != nil {
		t.Fatalf("trainFrequencyModel vrací chybu: %v", err)
	}

	if len(frequency) == 0 {
		t.Fatalf("trainFrequencyModel vrátil prázdný model")
	}
}

func BenchmarkCorrectSentenceByVariants(b *testing.B) {
	frequency := loadRealFrequencyModel(b)

	for i := 0; i < b.N; i++ {
		_ = correctSentenceByVariants(benchmarkSentence, frequency)
	}
}

func BenchmarkCorrectSentenceByDictionaryDistance(b *testing.B) {
	frequency := loadRealFrequencyModel(b)

	for i := 0; i < b.N; i++ {
		_ = correctSentenceByDictionaryDistance(benchmarkSentence, frequency)
	}
}
