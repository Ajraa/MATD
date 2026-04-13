package main

import (
	"math/bits"
	"strings"
)

// UNÁRNÍ KÓDOVÁNÍ
// UnaryEncode zakóduje kladné číslo n do unárního kódu.
func UnaryEncode(n int) string {
	if n <= 0 {
		panic("unární kód je definován pouze pro n >= 1")
	}
	return strings.Repeat("1", n-1) + "0"
}

// UnaryDecode dekóduje unární kód ze začátku řetězce, vrátí hodnotu a zbytek.
func UnaryDecode(code string) (int, string) {
	count := 0
	for i := 0; i < len(code); i++ {
		if code[i] == '1' {
			count++
		} else {
			// narazili jsme na '0' — konec kódového slova
			return count + 1, code[i+1:]
		}
	}
	panic("neplatný unární kód: chybí ukončovací 0")
}

// UnaryEncodeList zakóduje seznam kladných čísel do jednoho řetězce.
func UnaryEncodeList(nums []int) string {
	var sb strings.Builder
	for _, n := range nums {
		sb.WriteString(UnaryEncode(n))
	}
	return sb.String()
}

// UnaryDecodeList dekóduje celý řetězec zpět na seznam čísel.
func UnaryDecodeList(code string) []int {
	var result []int
	for len(code) > 0 {
		val, rest := UnaryDecode(code)
		result = append(result, val)
		code = rest
	}
	return result
}

// ELIASOVO GAMMA KÓDOVÁNÍ
// binFixed vrátí binární reprezentaci val na přesně width číslic.
func binFixed(width, val int) string {
	b := make([]byte, width)
	for i := width - 1; i >= 0; i-- {
		if val&1 == 1 {
			b[i] = '1'
		} else {
			b[i] = '0'
		}
		val >>= 1
	}
	return string(b)
}

// EliasGammaEncode zakóduje kladné číslo n Eliasovým gamma kódem.
func EliasGammaEncode(n int) string {
	if n <= 0 {
		panic("Eliasovo kódování je definováno pouze pro n >= 1")
	}
	k := bits.Len(uint(n)) - 1 // floor(log2(n))
	prefix := strings.Repeat("0", k) + "1"
	suffix := ""
	if k > 0 {
		mask := (1 << k) - 1
		suffix = binFixed(k, n&mask)
	}
	return prefix + suffix
}

// EliasGammaDecode dekóduje gamma kód ze začátku řetězce, vrátí hodnotu a zbytek.
func EliasGammaDecode(code string) (int, string) {
	k := 0
	for k < len(code) && code[k] == '0' {
		k++
	}
	if k >= len(code) || code[k] != '1' {
		panic("neplatný Eliasův gamma kód: chybí ukončovací 1")
	}
	pos := k + 1
	if pos+k > len(code) {
		panic("neplatný Eliasův gamma kód: kód je příliš krátký")
	}
	// vedoucí bit je 2^k; suffix bits jsou bity k-1 dolů do 0
	val := 1 << k
	for i := 0; i < k; i++ {
		if code[pos+i] == '1' {
			val |= (1 << (k - 1 - i))
		}
	}
	return val, code[pos+k:]
}

// EliasGammaEncodeList zakóduje seznam čísel.
func EliasGammaEncodeList(nums []int) string {
	var sb strings.Builder
	for _, n := range nums {
		sb.WriteString(EliasGammaEncode(n))
	}
	return sb.String()
}

// EliasGammaDecodeList dekóduje celý řetězec zpět na seznam čísel.
func EliasGammaDecodeList(code string) []int {
	var result []int
	for len(code) > 0 {
		val, rest := EliasGammaDecode(code)
		result = append(result, val)
		code = rest
	}
	return result
}

// FIBONACCIHO KÓDOVÁNÍ
// fibTable obsahuje předpočítaná Fibonacciho čísla (F1=1, F2=2, F3=3, F4=5, ...).
// Generuje všechna čísla, která se vejdou do int — stačí pro libovolnou praktickou hodnotu.
var fibTable = func() []int {
	t := make([]int, 0, 50)
	a, b := 1, 2
	for a > 0 { // zastaví se při přetečení int
		t = append(t, a)
		a, b = b, a+b
	}
	return t
}()

// FibonacciEncode zakóduje kladné číslo n Fibonacciho kódem.
func FibonacciEncode(n int) string {
	if n <= 0 {
		panic("Fibonacciho kódování je definováno pouze pro n >= 1")
	}
	k := 0
	for k < len(fibTable) && fibTable[k] <= n {
		k++
	}
	bitsArr := make([]byte, k)
	rem := n
	for i := k - 1; i >= 0; i-- {
		if fibTable[i] <= rem {
			bitsArr[i] = '1'
			rem -= fibTable[i]
		} else {
			bitsArr[i] = '0'
		}
	}
	return string(bitsArr) + "1"
}

// FibonacciDecode dekóduje Fibonacciho kód ze začátku řetězce, vrátí hodnotu a zbytek.
func FibonacciDecode(code string) (int, string) {
	// hledáme první výskyt "11" (terminační konec kódového slova)
	end := -1
	for i := 0; i < len(code)-1; i++ {
		if code[i] == '1' && code[i+1] == '1' {
			end = i + 1
			break
		}
	}
	if end == -1 {
		panic("neplatný Fibonacciho kód: chybí terminační '11'")
	}
	val := 0
	for i := 0; i < end; i++ {
		if code[i] == '1' {
			val += fibTable[i]
		}
	}
	return val, code[end+1:]
}

// FibonacciEncodeList zakóduje seznam čísel.
func FibonacciEncodeList(nums []int) string {
	var sb strings.Builder
	for _, n := range nums {
		sb.WriteString(FibonacciEncode(n))
	}
	return sb.String()
}

// FibonacciDecodeList dekóduje celý řetězec zpět na seznam čísel.
func FibonacciDecodeList(code string) []int {
	var result []int
	for len(code) > 0 {
		val, rest := FibonacciDecode(code)
		result = append(result, val)
		code = rest
	}
	return result
}

// POMOCNÉ FUNKCE PRO GAP ENCODING
// ToGaps převede seřazený seznam docIDs na seznam rozdílů (gaps).
// První prvek = první docID (rozdíl od 0).
// Příklad: [3, 7, 10] → [3, 4, 3]
func ToGaps(docIDs []int) []int {
	if len(docIDs) == 0 {
		return nil
	}
	gaps := make([]int, len(docIDs))
	gaps[0] = docIDs[0]
	for i := 1; i < len(docIDs); i++ {
		gaps[i] = docIDs[i] - docIDs[i-1]
	}
	return gaps
}

// FromGaps rekonstruuje seřazený seznam docIDs ze seznamu rozdílů.
func FromGaps(gaps []int) []int {
	if len(gaps) == 0 {
		return nil
	}
	docIDs := make([]int, len(gaps))
	docIDs[0] = gaps[0]
	for i := 1; i < len(gaps); i++ {
		docIDs[i] = docIDs[i-1] + gaps[i]
	}
	return docIDs
}
