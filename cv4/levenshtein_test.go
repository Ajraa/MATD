package main

import "testing"

func TestLevenshteinDistance(t *testing.T) {
	tests := []struct {
		name     string
		s1, s2   string
		expected int
	}{
		// Prázdné řetězce
		{"oba prázdné", "", "", 0},
		{"první prázdný", "", "abc", 3},
		{"druhý prázdný", "abc", "", 3},

		// Shodné řetězce
		{"stejná slova", "hello", "hello", 0},
		{"jeden znak stejný", "a", "a", 0},

		// Pouze vložení
		{"jedno vložení", "cat", "cats", 1},
		{"vložení na začátek", "at", "cat", 1},

		// Pouze smazání
		{"jedno smazání", "cats", "cat", 1},
		{"smazání ze začátku", "cat", "at", 1},

		// Pouze substituce
		{"jedna substituce", "cat", "car", 1},
		{"substituce uprostřed", "bat", "bot", 1},

		// Klasické příklady
		{"kitten→sitting", "kitten", "sitting", 3},
		{"saturday→sunday", "saturday", "sunday", 3},
		{"intention→execution", "intention", "execution", 5},
		{"rosettacode→raisethysword", "rosettacode", "raisethysword", 8},

		// Kompletní výměna
		{"kompletní výměna krátkých", "abc", "xyz", 3},

		// Různé délky
		{"různá délka", "a", "abcdef", 5},
		{"opačně různá délka", "abcdef", "a", 5},

		// Jeden znak
		{"jeden znak rozdíl", "a", "b", 1},

		// Diakritika (UTF-8 – funkce pracuje s bajty, ne runami)
		{"ascii bez diakritiky", "rok", "bok", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := levenshteinDistance(tt.s1, tt.s2)
			if got != tt.expected {
				t.Errorf("levenshteinDistance(%q, %q) = %d, očekáváno %d",
					tt.s1, tt.s2, got, tt.expected)
			}
		})
	}
}
