package main

import (
	"fmt"
	"os"
)

func main() {
	path := "C:\\Users\\ajrac\\Downloads\\cs (1).txt\\cs (1).txt"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	fmt.Println("Using path:", path)

	data, err := os.ReadFile(path)

	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

}

func KMP(T, P string) []int {
	m := len(T)
	n := len(P)

	lsp := make([]int, m)

	i := 0
	j := 0
	for i < n {
		if T[i] == P[j] {
			i++
			j++
		}

		if j == n {
			// shoda
			j = lsp[j-1]
		}
	}
}

func horspool(T, P string) []int {
	m := len(T)
	n := len(P)

	shift := makeShiftTable(P, n)
	ret := make([]int, 0)

	i := m - 1

	for i < m {
		k := 0
		for k < n && P[m-1-k] == T[i-k] {
			k++
		}

		if k == n {
			// shoda
			ret = append(ret, i-k+1)
		}

		i += shift[T[i]]
	}
	return ret
}

func makeShiftTable(P string, n int) map[byte]int {
	shift := make(map[byte]int)
	for i := 0; i < n-1; i++ {
		shift[P[i]] = n - 1 - i
	}
	return shift
}
