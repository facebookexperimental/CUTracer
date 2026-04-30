# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Generator, Optional

import yscope_clp_core  # type: ignore
from cutracer.query.reader import TraceReaderBase

# pyrefly: ignore [missing-import]
from yscope_clp_core import KqlQuery


class TraceReaderCLP(TraceReaderBase):
    """
    Reader for CUTracer trace files.

    Supports CLP format.
    Provides efficient iteration over trace records.

    Example:
        >>> reader = TraceReaderCLP("trace.clp")
        >>> for record in reader.iter_records():
        ...     print(record["sass"])
    """

    def __init__(self, file):
        assert file.exists(), f"Non-exist clp archive file: {file.absolute()}"
        self._archive = file

    def _filter_expr_to_clp_query(self, filter_exprs: Optional[tuple[str, ...]]) -> str:
        if not filter_exprs:
            return "*"
        result = []
        for filter_expr in filter_exprs:
            filter_array = filter_expr.split(";")
            for filter_clause in filter_array:
                if "=" not in filter_clause:
                    raise ValueError(
                        f"Invalid filter expression: '{filter_clause}'. Expected format: 'field=value'"
                    )
                lhs, _sym, rhs = filter_clause.partition("=")
                result.append((lhs, rhs))
        result_str = " AND ".join(f"{lhs}: {rhs}" for lhs, rhs in result)
        if result_str:
            return result_str
        return "*"

    def iter_records(self, filter_exprs: Optional[tuple[str, ...]] = None) -> Generator:
        clp_query = KqlQuery(self._filter_expr_to_clp_query(filter_exprs))
        with yscope_clp_core.search_archive(self._archive, clp_query) as record_iter:
            for next_record in record_iter:
                yield next_record.get_kv_pairs()
