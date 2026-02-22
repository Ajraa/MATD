package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	path := "data.csv"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	fmt.Println("Using path:", path)

	data, err := os.ReadFile(path)

	if err != nil {
		fmt.Println("Error reading file:", err)
		return
	}

	fmt.Println("File size:", len(data), "bytes")

	text := clean(string(data))
	printStatistics(text)

	tok := ByteTokenizer{}
	vocab, sequence := tok.Tokenize(text, 1000)
	fmt.Println("Vocab size:", len(vocab))
	fmt.Println("Sequence length:", len(sequence))

}

func clean(text string) string {
	text = strings.ToLower(text)
	text = strings.Join(strings.Fields(text), " ")
	return text
}
