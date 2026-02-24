package main

import (
	"fmt"
	"strings"
)

func main() {
	// path := "data.csv"
	// if len(os.Args) > 1 {
	// 	path = os.Args[1]
	// }
	// fmt.Println("Using path:", path)

	// data, err := os.ReadFile(path)

	// if err != nil {
	// 	fmt.Println("Error reading file:", err)
	// 	return
	// }

	// fmt.Println("File size:", len(data), "bytes")

	text := `the cat sat on the mat the cat ate the rat and the bat sat on the flat hat ` +
		`the cat sat on the mat again and the rat ran from the bat the cat and the rat sat together ` +
		`on the mat while the bat flew over the flat hat the cat chased the rat around the mat and ` +
		`the bat watched from the hat`
	text = clean(text)
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
