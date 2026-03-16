package main

import "testing"

func TestEdits1ContainsAllOperations(t *testing.T) {
	alphabet := []rune{'a', 'b', 'c'}
	variants := edits1("ab", alphabet)

	cases := []string{
		"b",   // delete
		"ac",  // replace
		"aab", // insert
		"ba",  // transpose
	}

	for _, candidate := range cases {
		if _, ok := variants[candidate]; !ok {
			t.Fatalf("expected candidate %q to be present", candidate)
		}
	}
}

func TestEditsUpToDistance2IncludesDistance2Words(t *testing.T) {
	alphabet := []rune{'a', 'b', 'c'}
	variants := editsUpToDistance2("ab", alphabet)

	if _, ok := variants["cc"]; !ok {
		t.Fatalf("expected candidate %q to be present (distance 2)", "cc")
	}

	if _, ok := variants["ab"]; ok {
		t.Fatalf("original word must not be included")
	}
}
