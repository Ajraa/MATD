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

	tok := WordTokenizer{}
	tokens := tok.Tokenize(text, 1000)
	fmt.Println("Tokens:", tokens)

}

func clean(text string) string {
	text = strings.ToLower(text)
	text = strings.Join(strings.Fields(text), " ")
	return text
}
