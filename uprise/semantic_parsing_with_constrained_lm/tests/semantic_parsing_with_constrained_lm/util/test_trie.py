# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.trie import CompressedTrie, Trie


def powerset(lst):
    return itertools.chain.from_iterable(
        itertools.combinations(lst, r) for r in range(len(lst) + 1)
    )


def test_comprehensive():
    # Equal to [(), (0,), (1,), (0, 0), ..., (1, 1)]. Length 7.
    all_sequences = list(
        itertools.chain.from_iterable(
            itertools.product(range(2), repeat=length) for length in range(3)
        )
    )

    for subset in powerset(all_sequences):
        for permutation in itertools.permutations(subset):
            trie = Trie[int]()
            reference = set()
            for elem in permutation:
                trie.add(elem)
                reference.add(elem)

                assert set(trie) == reference
                assert len(trie) == len(reference)
                for other_elem in subset:
                    assert (other_elem in trie) == (other_elem in reference)


def test_prefix():
    trie = Trie[str](["ab", "abba", "abab", "abbaa"])
    assert set(trie.prefix_items("")) == {
        tuple(x) for x in ["ab", "abba", "abab", "abbaa"]
    }
    assert set(trie.prefix_items("a")) == {
        tuple(x) for x in ["ab", "abba", "abab", "abbaa"]
    }
    assert set(trie.prefix_items("ab")) == {
        tuple(x) for x in ["ab", "abba", "abab", "abbaa"]
    }
    assert set(trie.prefix_items("abb")) == {tuple(x) for x in ["abba", "abbaa"]}
    assert set(trie.prefix_items("b")) == set()
    assert set(trie.prefix_items("abaa")) == set()

    assert trie.prefix_count("") == 4
    assert trie.prefix_count("a") == 4
    assert trie.prefix_count("ab") == 4
    assert trie.prefix_count("abb") == 2
    assert trie.prefix_count("b") == 0
    assert trie.prefix_count("abaa") == 0


def test_compressed():
    def T(*elems):
        return set(tuple(elem) for elem in elems)

    ct = CompressedTrie(Trie(["abcd", "abcef"]))
    assert len(ct) == 2
    assert ct.prefix_next("") == (T("abc"), False)
    assert ct.prefix_next("abc") == (T("d", "ef"), False)
    assert ct.prefix_next("abcd") == (T(), True)
    assert ct.prefix_next("abcef") == (T(), True)

    ct = CompressedTrie(Trie(["a", "ab", "abc"]))
    assert len(ct) == 3
    assert ct.prefix_next("") == (T("a"), False)
    assert ct.prefix_next("a") == (T("b"), True)
    assert ct.prefix_next("ab") == (T("c"), True)
    assert ct.prefix_next("abc") == (T(), True)

    ct = CompressedTrie(Trie(["a", "ab", "abcd"]))
    assert len(ct) == 3
    assert ct.prefix_next("") == (T("a"), False)
    assert ct.prefix_next("a") == (T("b"), True)
    assert ct.prefix_next("ab") == (T("cd"), True)
    assert ct.prefix_next("abcd") == (T(), True)

    ct = CompressedTrie(Trie[str]())
    assert len(ct) == 0

    ct = CompressedTrie(Trie([""]))
    assert len(ct) == 1


def test_compressed_comprehensive():
    all_sequences = list(
        itertools.chain.from_iterable(
            itertools.product(range(2), repeat=length) for length in range(3)
        )
    )

    for subset in powerset(all_sequences):
        ct = CompressedTrie(Trie(subset))
        assert set(iter(ct)) == set(subset)
        assert len(ct) == len(subset)
        for elem in subset:
            assert (elem in ct) == (elem in subset)
