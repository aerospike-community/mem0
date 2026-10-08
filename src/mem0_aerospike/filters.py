try:
    from aerospike_sdk import Exp, ExpType, MapReturnType, StringWriteFlags
except ImportError as e:
    raise ImportError(
        "Aerospike vector store support requires the preview 'aerospike-sdk' package. "
        "Install it with: pip install 'aerospike-sdk'."
    ) from e

# Fixed core-bin map: these payload keys get dedicated bins. Every other
# payload key is bundled into the single `metadata` Map (CDT) bin, since
# Aerospike bin names are capped at 15 characters server-side.
CORE_STRING_FIELDS = ("data", "hash", "user_id", "agent_id", "run_id", "text_lemmatized", "attributed_to")
TIMESTAMP_FIELDS = ("created_at", "updated_at")
SCOPE_FIELDS = ("user_id", "agent_id", "run_id")
SUPPORTED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"}

EMBEDDING_BIN = "embedding"
METADATA_BIN = "metadata"


def _exp_type_for(value):
    if isinstance(value, bool):
        return ExpType.BOOL
    if isinstance(value, int):
        return ExpType.INT
    if isinstance(value, float):
        return ExpType.FLOAT
    if isinstance(value, str):
        return ExpType.STRING
    raise ValueError(f"Unsupported filter value type: {type(value).__name__}")


def _field_expr(key: str, sample_value):
    """Build the Exp for a field: a core bin, a timestamp bin, or a metadata map key."""
    if key in CORE_STRING_FIELDS:
        return Exp.string_bin(key)
    if key in TIMESTAMP_FIELDS:
        return Exp.int_bin(f"{key}_ms")
    value_type = _exp_type_for(sample_value)
    return Exp.map_get_by_key(MapReturnType.VALUE, value_type, Exp.string_val(key), Exp.map_bin(METADATA_BIN), [])


def _compile_operator(key: str, op: str, value):
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator '{op}' for field '{key}'. Supported operators: {sorted(SUPPORTED_OPERATORS)}"
        )
    if op == "eq":
        return Exp.eq(_field_expr(key, value), Exp.val(value))
    if op == "ne":
        return Exp.ne(_field_expr(key, value), Exp.val(value))
    if op == "gt":
        return Exp.gt(_field_expr(key, value), Exp.val(value))
    if op == "gte":
        return Exp.ge(_field_expr(key, value), Exp.val(value))
    if op == "lt":
        return Exp.lt(_field_expr(key, value), Exp.val(value))
    if op == "lte":
        return Exp.le(_field_expr(key, value), Exp.val(value))
    if op in ("in", "nin"):
        values = list(value)
        if not values:
            return Exp.val(op == "nin")
        sample = values[0]
        expr = Exp.in_list(_field_expr(key, sample), Exp.val(values))
        return Exp.not_(expr) if op == "nin" else expr
    if op == "contains":
        return Exp.string_contains(Exp.string_val(str(value)), _field_expr(key, value))
    # icontains
    lowered_field = Exp.string_lower(int(StringWriteFlags.DEFAULT), _field_expr(key, value))
    return Exp.string_contains(Exp.string_val(str(value).lower()), lowered_field)


def _compile_field(key: str, condition):
    """Build the Exp for one field: literal equality, or an operator dict."""
    if not isinstance(condition, dict):
        return Exp.eq(_field_expr(key, condition), Exp.val(condition))
    exprs = [_compile_operator(key, op, value) for op, value in condition.items()]
    return exprs[0] if len(exprs) == 1 else Exp.and_(exprs)


def compile_filters(filters, exclude_key=None):
    """Compile a mem0 filter dict into an Exp tree. Never interpolates values into AEL strings.

    Recurses into itself for $or/$not sub-conditions, so a sub-condition may
    itself contain nested $or/$not/field clauses.
    """
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if key == exclude_key:
            continue
        if key == "$or":
            conditions = value if isinstance(value, list) else [value]
            sub = [c for c in (compile_filters(cond, exclude_key=exclude_key) for cond in conditions) if c is not None]
            if sub:
                clauses.append(sub[0] if len(sub) == 1 else Exp.or_(sub))
        elif key == "$not":
            conditions = value if isinstance(value, list) else [value]
            for cond in conditions:
                compiled = compile_filters(cond, exclude_key=exclude_key)
                if compiled is not None:
                    clauses.append(Exp.not_(compiled))
        else:
            clauses.append(_compile_field(key, value))
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else Exp.and_(clauses)
