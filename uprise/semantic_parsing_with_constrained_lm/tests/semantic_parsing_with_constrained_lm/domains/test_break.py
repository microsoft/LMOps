# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import GPT2Tokenizer

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.qdmr_break import (
    BreakCommonTokens,
    BreakDataType,
    BreakDatum,
    BreakPartialParse,
    BreakPieces,
    can_close_paren,
)


def test_nest():
    s = "return war missions ;return #1 of england ;return #2 in world war 2"

    assert BreakPieces.nest(s) == "( ( war missions ) of england ) in world war 2"

    s = ""

    assert BreakPieces.nest(s) == ""


def test_nest2():
    s = "return soda bottles ;return #1 that are large ;return brand of #2 ;return #2 where  #3 is the  same ;return cap of #4 ;return label of #4 ;return colors of #5 ;return colors of #6 ;return #4 where  #7 are the  same ;return #4 where  #8 are the  same ;return #4 in  both  #9 and #10 ;return design details of #11 ;return #11 where  #12 are different ;return number of  #13 ;return if  #14 is equal to  two"

    assert BreakPieces.nest(s) == (
        "if ( number of ( ( ( ( ( soda bottles ) that are large ) where ( brand of ( "
        "( soda bottles ) that are large ) ) is the same ) in both ( ( ( ( soda "
        "bottles ) that are large ) where ( brand of ( ( soda bottles ) that are "
        "large ) ) is the same ) where ( colors of ( cap of ( ( ( soda bottles ) that "
        "are large ) where ( brand of ( ( soda bottles ) that are large ) ) is the "
        "same ) ) ) are the same ) and ( ( ( ( soda bottles ) that are large ) where "
        "( brand of ( ( soda bottles ) that are large ) ) is the same ) where ( "
        "colors of ( label of ( ( ( soda bottles ) that are large ) where ( brand of "
        "( ( soda bottles ) that are large ) ) is the same ) ) ) are the same ) ) "
        "where ( design details of ( ( ( ( soda bottles ) that are large ) where ( "
        "brand of ( ( soda bottles ) that are large ) ) is the same ) in both ( ( ( ( "
        "soda bottles ) that are large ) where ( brand of ( ( soda bottles ) that are "
        "large ) ) is the same ) where ( colors of ( cap of ( ( ( soda bottles ) that "
        "are large ) where ( brand of ( ( soda bottles ) that are large ) ) is the "
        "same ) ) ) are the same ) and ( ( ( ( soda bottles ) that are large ) where "
        "( brand of ( ( soda bottles ) that are large ) ) is the same ) where ( "
        "colors of ( label of ( ( ( soda bottles ) that are large ) where ( brand of "
        "( ( soda bottles ) that are large ) ) is the same ) ) ) are the same ) ) ) "
        "are different ) ) is equal to two"
    )


def test_nest3():
    s = "return abraham lincoln ; return political party of #1"

    assert BreakPieces.nest(s) == "political party of ( abraham lincoln )"


def test_unnest():
    s = "the difference of 100 and ( A ) , ( the sum of ( B ) )"

    assert (
        BreakPieces.unnest(s)
        == "return B ;return A ;return the sum of #1 ;return the difference of 100 and #2 , #3"
    )

    s = ""

    assert BreakPieces.unnest(s) == "return "


def test_simplify():
    s = "( ( A ) ), ( ( B ) )"
    assert BreakPieces.simplify_nesting(s) == "( A ), ( B )"

    s = "( A )"
    assert BreakPieces.simplify_nesting(s) == "A"

    s = "( A ) of ( B )"
    assert BreakPieces.simplify_nesting(s) == "( A ) of ( B )"

    s = "( ( A ) of B )"
    assert BreakPieces.simplify_nesting(s) == "( A ) of B"


def test_is_balanced():
    is_balanced, max_depth = BreakPieces.is_balanced("(()) ((()))")

    assert is_balanced
    assert max_depth == 3


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
common_tokens = BreakCommonTokens.from_tokenizer(tokenizer)


def test_can_close_paren():
    assert not can_close_paren("A")

    assert can_close_paren("( A")
    assert can_close_paren("( A B")

    assert can_close_paren("( ( A")
    assert can_close_paren("( ( A )")

    assert can_close_paren("( ( A ) of ( B )")
    assert not can_close_paren("( ( A ) of ( B ) )")

    assert can_close_paren("Z ( ( A")
    assert can_close_paren("Z ( ( A )")
    assert not can_close_paren("Z ( ( A ) )")

    assert can_close_paren("Z ( ( A ) of ( B )")
    assert not can_close_paren("Z ( ( A ) of ( B ) )")


def test_possible_next_tokens():
    # {next_token_id} is allowed_tokens
    [next_token_id] = tokenizer.encode("next", add_special_tokens=False)
    datum = BreakDatum(None, None, None, "", "", {next_token_id}, "", "",)
    pp = BreakPartialParse.initial(common_tokens, BreakDataType.nested, datum)

    def advance(text: str):
        new_pp = pp
        for token in tokenizer.encode(text, add_special_tokens=False):
            new_pp = new_pp.append(token)
        return new_pp

    next_tokens, is_eos = advance(" hello").allowed_next()

    assert set(next_tokens.tolist()) == {next_token_id, common_tokens.open_paren}
    assert is_eos

    next_tokens, is_eos = advance(" ( hello").allowed_next()
    assert set(next_tokens.tolist()) == {
        next_token_id,
        common_tokens.open_paren,
        common_tokens.close_paren,
    }
    assert not is_eos

    next_tokens, is_eos = advance(" ( hello )").allowed_next()

    assert set(next_tokens.tolist()) == {next_token_id}
    assert is_eos
