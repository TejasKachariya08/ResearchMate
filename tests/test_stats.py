"""Tests for stats and introspection in ResearchMate."""

from unittest.mock import MagicMock
from researchmate.stats import (
    build_attribute_rows,
    build_index_rows,
    build_stats_rows,
    get_index_info,
)


def test_get_index_info_calls_redis():
    mock_redis = MagicMock()
    mock_redis.ft.return_value.info.return_value = {"index_name": "researchmate_quantum"}

    res = get_index_info("researchmate_quantum", redis_client=mock_redis)

    assert res["index_name"] == "researchmate_quantum"
    mock_redis.ft.assert_called_once_with("researchmate_quantum")


def test_build_index_rows():
    info = {
        "index_name": "researchmate_llms",
        "index_definition": ["key_type", "HASH", "prefixes", ["researchmate:doc:llms:"]],
        "index_options": [],
        "indexing": 0,
    }
    rows = build_index_rows(info)
    assert len(rows) == 1
    assert rows[0]["Index Name"] == "researchmate_llms"
    assert rows[0]["Storage Type"] == "HASH"


def test_build_attribute_rows():
    info = {
        "attributes": [
            [
                "identifier", "embedding",
                "attribute", "embedding",
                "type", "VECTOR",
                "algorithm", "FLAT",
                "dim", 3072,
            ]
        ]
    }
    rows = build_attribute_rows(info)
    assert len(rows) == 1
    assert rows[0]["Identifier"] == "embedding"
    assert rows[0]["Type"] == "VECTOR"
    assert "algorithm=FLAT" in rows[0]["Configuration"]


def test_build_stats_rows():
    info = {
        "num_docs": 12,
        "num_records": 12,
        "vector_index_sz_mb": 0.45,
    }
    rows = build_stats_rows(info)
    stat_dict = {r["Metric"]: r["Value"] for r in rows}

    assert stat_dict["num_docs"] == 12
    assert stat_dict["vector_index_sz_mb"] == 0.45
