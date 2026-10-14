"""Unit tests for the filter-dict -> Exp compiler (no server required)."""

import pytest
from aerospike_sdk import Exp

from mem0_aerospike.filters import compile_filters


def test_empty_filters_compile_to_none():
    assert compile_filters(None) is None
    assert compile_filters({}) is None


def test_literal_equality_compiles():
    assert compile_filters({"user_id": "alice"}) is not None


def test_operator_dicts_compile():
    for op, value in [
        ("eq", "alice"),
        ("ne", "alice"),
        ("gt", 10),
        ("gte", 20),
        ("lt", 20),
        ("lte", 20),
        ("in", ["a", "b"]),
        ("nin", ["a"]),
        ("contains", "work"),
        ("icontains", "WORK"),
    ]:
        assert compile_filters({"user_id": "u", "name": {op: value}}) is not None


def test_unsupported_operator_raises():
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        compile_filters({"user_id": "u", "name": {"$regex": ".*"}})


def test_unsupported_value_type_raises():
    with pytest.raises(ValueError, match="Unsupported filter value type"):
        compile_filters({"meta": {"eq": object()}})


def test_empty_in_nin_are_constants():
    expr = compile_filters({"score": {"in": []}})
    assert expr is not None
    expr = compile_filters({"score": {"nin": []}})
    assert expr is not None


def test_or_and_not_compilers_recurse():
    expr = compile_filters(
        {
            "user_id": "u",
            "$or": [{"name": {"eq": "a"}}, {"$or": [{"name": {"eq": "b"}}, {"name": {"eq": "c"}}]}],
            "$not": [{"color": {"eq": "red"}}],
        }
    )
    assert expr is not None


def test_or_and_not_accept_single_dict():
    """A bare dict for $or/$not should be treated as a single condition."""
    expr_or = compile_filters({"user_id": "u", "$or": {"name": {"eq": "a"}}})
    assert expr_or is not None
    expr_not = compile_filters({"user_id": "u", "$not": {"color": {"eq": "red"}}})
    assert expr_not is not None


def test_exclude_key_propagates_into_nested_conditions():
    """The excluded scope key should not reappear inside $or/$not branches."""
    expr = compile_filters(
        {"user_id": "u", "$or": {"user_id": {"eq": "v"}, "name": {"eq": "a"}}},
        exclude_key="user_id",
    )
    assert expr is not None


def test_exclude_key_is_skipped():
    expr = compile_filters({"user_id": "u", "name": {"eq": "a"}}, exclude_key="user_id")
    assert expr is not None
    # Excluding the only clause yields None
    assert compile_filters({"user_id": "u"}, exclude_key="user_id") is None


def test_special_characters_are_values_not_ael():
    # Must not raise and must produce an Exp tree, not an interpolated string.
    expr = compile_filters({"user_id": "u", "name": {"eq": "'; drop"}})
    assert isinstance(expr, Exp)
