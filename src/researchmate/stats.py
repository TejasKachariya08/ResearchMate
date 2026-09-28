"""Redis Vector Index statistics and introspection module for ResearchMate."""

from __future__ import annotations

from typing import Any, List, Dict
from redis import Redis
from researchmate.config import get_settings


STATS_KEYS = [
    "num_docs",
    "num_records",
    "number_of_uses",
    "percent_indexed",
    "total_indexing_time",
    "bytes_per_record_avg",
    "records_per_doc_avg",
    "doc_table_size_mb",
    "vector_index_sz_mb",
]


def get_stats_redis_client(redis_url: str | None = None) -> Redis:
    """Get Redis client for string-decoded queries."""
    url = redis_url or get_settings().redis_url
    return Redis.from_url(url, decode_responses=True)


def get_index_info(
    index_name: str,
    redis_url: str | None = None,
    redis_client: Redis | None = None,
) -> Dict[str, Any]:
    """Retrieve full FT.INFO dictionary for a given index."""
    client = redis_client or get_stats_redis_client(redis_url)
    try:
        return client.ft(index_name).info()
    finally:
        if redis_client is None:
            try:
                client.close()
            except Exception:
                pass


def _pairs_to_dict(value: Any) -> Dict[str, Any]:
    """Convert flat alternating key-value lists into python dictionaries."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, (list, tuple)):
        return {}

    result: Dict[str, Any] = {}
    iterator = iter(value)
    for key in iterator:
        result[str(key)] = next(iterator, None)
    return result


def _normalize_stat_values_for_table(values: List[Any]) -> List[Any]:
    """Normalize types so Streamlit tables render cleanly without PyArrow mixed-type errors."""
    non_null_values = [v for v in values if v is not None]
    has_string = any(isinstance(v, str) for v in non_null_values)
    has_numeric = any(isinstance(v, (int, float)) for v in non_null_values)
    has_other = any(not isinstance(v, (str, int, float)) for v in non_null_values)

    if has_other or (has_string and has_numeric):
        return [None if v is None else str(v) for v in values]

    return values


def build_stats_rows(index_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a structured list of key performance and storage metrics."""
    values = _normalize_stat_values_for_table([index_info.get(key) for key in STATS_KEYS])
    return [{"Metric": key, "Value": value} for key, value in zip(STATS_KEYS, values)]


def build_index_rows(index_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build summary rows describing the index definition."""
    definition = _pairs_to_dict(index_info.get("index_definition", {}))
    return [
        {
            "Index Name": index_info.get("index_name"),
            "Storage Type": definition.get("key_type"),
            "Prefixes": str(definition.get("prefixes")),
            "Index Options": str(index_info.get("index_options")),
            "Indexing Active": str(index_info.get("indexing")),
        }
    ]


def build_attribute_rows(index_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build rows describing each field and its vector/text configuration in the schema."""
    rows: List[Dict[str, Any]] = []
    for attribute in index_info.get("attributes", []):
        attr_map = _pairs_to_dict(attribute)
        extras = [
            f"{k}={v}"
            for k, v in attr_map.items()
            if k not in {"identifier", "attribute", "type"}
        ]
        rows.append(
            {
                "Identifier": attr_map.get("identifier"),
                "Attribute": attr_map.get("attribute"),
                "Type": attr_map.get("type"),
                "Configuration": ", ".join(extras),
            }
        )
    return rows
