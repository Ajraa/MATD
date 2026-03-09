package main

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
	s1 := "kitten"
	s2 := "sitting"
	distance := levenshteinDistance(s1, s2)
	println("Levenshtein Distance:", distance)
}
