package main

import (
	"fmt"
	"sort"
	"strings"
)

// Množinové operace nad výsledky dotazů.

func intersect(a, b map[int]struct{}) map[int]struct{} {
	result := make(map[int]struct{})
	for id := range a {
		if _, ok := b[id]; ok {
			result[id] = struct{}{}
		}
	}
	return result
}

func union(a, b map[int]struct{}) map[int]struct{} {
	result := make(map[int]struct{}, len(a)+len(b))
	for id := range a {
		result[id] = struct{}{}
	}
	for id := range b {
		result[id] = struct{}{}
	}
	return result
}

func difference(universe, exclude map[int]struct{}) map[int]struct{} {
	result := make(map[int]struct{})
	for id := range universe {
		if _, ok := exclude[id]; !ok {
			result[id] = struct{}{}
		}
	}
	return result
}

// Rekurzivní sestupný parser boolean dotazů.
// Gramatika:
//
//	expr   = term { OR term }
//	term   = factor { AND factor }
//	factor = NOT factor | '(' expr ')' | word
type parser struct {
	tokens []string
	pos    int
	idx    *InvertedIndex
}

func newParser(query string, idx *InvertedIndex) *parser {
	raw := strings.FieldsFunc(query, func(r rune) bool {
		return r == ' ' || r == '\t'
	})
	var tokens []string
	for _, tok := range raw {
		tokens = append(tokens, splitParens(tok)...)
	}
	return &parser{tokens: tokens, idx: idx}
}

func splitParens(tok string) []string {
	var result []string
	cur := ""
	for _, r := range tok {
		if r == '(' || r == ')' {
			if cur != "" {
				result = append(result, cur)
				cur = ""
			}
			result = append(result, string(r))
		} else {
			cur += string(r)
		}
	}
	if cur != "" {
		result = append(result, cur)
	}
	return result
}

func (p *parser) peek() string {
	if p.pos >= len(p.tokens) {
		return ""
	}
	return p.tokens[p.pos]
}

func (p *parser) consume() string {
	tok := p.peek()
	p.pos++
	return tok
}

func (p *parser) parseExpr() map[int]struct{} {
	left := p.parseTerm()
	for strings.EqualFold(p.peek(), "OR") {
		p.consume()
		right := p.parseTerm()
		left = union(left, right)
	}
	return left
}

func (p *parser) parseTerm() map[int]struct{} {
	left := p.parseFactor()
	for strings.EqualFold(p.peek(), "AND") {
		p.consume()
		right := p.parseFactor()
		left = intersect(left, right)
	}
	return left
}

func (p *parser) parseFactor() map[int]struct{} {
	tok := p.peek()
	if strings.EqualFold(tok, "NOT") {
		p.consume()
		operand := p.parseFactor()
		return difference(p.idx.allDocs(), operand)
	}
	if tok == "(" {
		p.consume()
		result := p.parseExpr()
		if p.peek() == ")" {
			p.consume()
		}
		return result
	}
	p.consume()
	return p.idx.lookup(tok)
}

// evalQuery vyhodnotí boolean dotaz a vrátí seřazený seznam DocID.
func evalQuery(query string, idx *InvertedIndex) ([]int, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("prázdný dotaz")
	}
	p := newParser(query, idx)
	resultSet := p.parseExpr()
	ids := make([]int, 0, len(resultSet))
	for id := range resultSet {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids, nil
}
