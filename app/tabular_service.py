from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import re
from typing import Any, Iterator
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook

from .tool_runtime import ToolContext
from .workspace_service import TABULAR_EXTENSIONS, WorkspaceError, WorkspaceService


ALLOWED_FILTER_OPERATORS = {
    "eq",
    "ne",
    "contains",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "is_empty",
    "not_empty",
}
ALLOWED_AGGREGATIONS = {
    "count_rows",
    "count_nonempty",
    "sum",
    "average",
    "min",
    "max",
    "distinct_count",
}


class TabularError(WorkspaceError):
    pass


class TabularService:
    def __init__(
        self,
        workspace: WorkspaceService,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        config = settings.get("workspace") or {}
        self.max_scan_rows = min(
            200_000, max(100, int(config.get("max_table_scan_rows", 20_000)))
        )
        self.max_rows_per_call = min(
            2_000, max(10, int(config.get("max_table_rows_per_call", 200)))
        )
        self.max_columns = min(
            300, max(5, int(config.get("max_table_columns", 100)))
        )
        self.max_groups = min(
            2_000, max(10, int(config.get("max_table_groups", 200)))
        )
        self.max_cell_chars = min(
            20_000, max(100, int(config.get("max_table_cell_chars", 2_000)))
        )
        self.max_workbook_sheets = min(
            200, max(1, int(config.get("max_workbook_sheets", 50)))
        )
        self.max_xlsx_uncompressed_bytes = min(
            500_000_000,
            max(
                5_000_000,
                int(config.get("max_xlsx_uncompressed_bytes", 100_000_000)),
            ),
        )

    def _path(self, path_value: str) -> Path:
        try:
            path = self.workspace._resolve(path_value, require_file=True)
            self.workspace._validate_file(path)
        except WorkspaceError as exc:
            raise TabularError(str(exc)) from exc
        if path.suffix.lower() not in TABULAR_EXTENSIONS:
            raise TabularError("该文件不是支持的CSV、TSV或XLSX表格")
        return path

    @staticmethod
    def _decode_text(path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            try:
                return raw.decode("gb18030"), "gb18030"
            except UnicodeDecodeError as exc:
                raise TabularError("表格文本编码不是UTF-8或GB18030") from exc

    def _validate_xlsx_archive(self, path: Path) -> None:
        try:
            with ZipFile(path) as archive:
                total = 0
                for item in archive.infolist():
                    total += item.file_size
                    if total > self.max_xlsx_uncompressed_bytes:
                        raise TabularError("XLSX解压后体积超过安全上限")
                    if (
                        item.file_size > 1_000_000
                        and item.compress_size > 0
                        and item.file_size / item.compress_size > 1_000
                    ):
                        raise TabularError("XLSX包含异常高压缩比内容，已拒绝读取")
        except BadZipFile as exc:
            raise TabularError("XLSX文件结构无效") from exc

    def _headers(self, values: list[Any]) -> list[str]:
        if len(values) > self.max_columns:
            raise TabularError(
                f"表格列数超过上限：{len(values)} > {self.max_columns}"
            )
        result: list[str] = []
        seen: dict[str, int] = {}
        for index, value in enumerate(values, start=1):
            base = str(value).strip() if value is not None else ""
            base = base[:200] or f"column_{index}"
            seen[base] = seen.get(base, 0) + 1
            result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        if not result:
            raise TabularError("表格没有可读取的列")
        return result

    def _cell(self, value: Any) -> tuple[Any, bool]:
        if value is None:
            return None, False
        if isinstance(value, (datetime, date, time)):
            return value.isoformat(), False
        if isinstance(value, Decimal):
            value = float(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None, False
        if isinstance(value, (bool, int, float)):
            return value, False
        text = str(value)
        return text[: self.max_cell_chars], len(text) > self.max_cell_chars

    @contextmanager
    def _source(
        self, path: Path, sheet: str | None = None
    ) -> Iterator[tuple[dict[str, Any], list[str], Iterator[tuple[int, list[Any]]]]]:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            text, encoding = self._decode_text(path)
            delimiter = "\t" if suffix == ".tsv" else ","
            if suffix == ".csv":
                try:
                    delimiter = csv.Sniffer().sniff(
                        text[:8192], delimiters=",;|\t"
                    ).delimiter
                except csv.Error:
                    pass
            lines = text.splitlines()
            reader = csv.reader(lines, delimiter=delimiter)
            try:
                headers = self._headers(next(reader))
            except StopIteration as exc:
                raise TabularError("表格为空") from exc

            def rows() -> Iterator[tuple[int, list[Any]]]:
                for row_number, row in enumerate(reader, start=2):
                    if len(row) > self.max_columns:
                        raise TabularError(
                            f"第{row_number}行列数超过上限{self.max_columns}"
                        )
                    yield row_number, row

            yield (
                {
                    "format": suffix[1:],
                    "encoding": encoding,
                    "delimiter": delimiter,
                    "sheet": None,
                    "sheets": [],
                    "estimated_rows": max(0, len(lines) - 1),
                },
                headers,
                rows(),
            )
            return

        self._validate_xlsx_archive(path)
        try:
            workbook = load_workbook(
                path, read_only=True, data_only=True, keep_links=False
            )
        except Exception as exc:
            raise TabularError(f"XLSX无法解析：{exc}") from exc
        try:
            if len(workbook.sheetnames) > self.max_workbook_sheets:
                raise TabularError(
                    f"工作表数量超过上限：{len(workbook.sheetnames)}"
                )
            selected = sheet or workbook.sheetnames[0]
            if selected not in workbook.sheetnames:
                raise TabularError("指定工作表不存在")
            worksheet = workbook[selected]
            if worksheet.max_column > self.max_columns:
                raise TabularError(
                    f"表格列数超过上限：{worksheet.max_column} > {self.max_columns}"
                )
            iterator = worksheet.iter_rows(values_only=True)
            try:
                headers = self._headers(list(next(iterator)))
            except StopIteration as exc:
                raise TabularError("工作表为空") from exc

            def rows() -> Iterator[tuple[int, list[Any]]]:
                for row_number, row in enumerate(iterator, start=2):
                    yield row_number, list(row)

            yield (
                {
                    "format": "xlsx",
                    "encoding": None,
                    "delimiter": None,
                    "sheet": selected,
                    "sheets": list(workbook.sheetnames),
                    "estimated_rows": max(0, worksheet.max_row - 1),
                    "formula_policy": "data_only；不执行公式，缺少缓存结果的公式单元格可能为空",
                },
                headers,
                rows(),
            )
        finally:
            workbook.close()

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    @staticmethod
    def _number(value: Any) -> Decimal | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            try:
                result = Decimal(str(value))
                return result if result.is_finite() else None
            except InvalidOperation:
                return None
        text = str(value).strip().replace(",", "")
        percent = text.endswith("%")
        if percent:
            text = text[:-1].strip()
        try:
            result = Decimal(text)
            if not result.is_finite():
                return None
            return result / 100 if percent else result
        except InvalidOperation:
            return None

    @staticmethod
    def _text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _compare(self, actual: Any, expected: Any) -> int:
        actual_number = self._number(actual)
        expected_number = self._number(expected)
        if actual_number is not None and expected_number is not None:
            return (actual_number > expected_number) - (actual_number < expected_number)
        left = self._text(actual).casefold()
        right = self._text(expected).casefold()
        return (left > right) - (left < right)

    def _matches(
        self, record: dict[str, Any], filters: list[dict[str, Any]]
    ) -> bool:
        for rule in filters:
            column = str(rule.get("column") or "")
            operator = str(rule.get("operator") or "")
            if column not in record:
                raise TabularError(f"筛选列不存在：{column}")
            if operator not in ALLOWED_FILTER_OPERATORS:
                raise TabularError(f"不支持的筛选操作：{operator}")
            actual = record[column]
            expected = rule.get("value")
            if operator == "is_empty":
                matched = self._is_empty(actual)
            elif operator == "not_empty":
                matched = not self._is_empty(actual)
            elif operator == "contains":
                matched = self._text(expected).casefold() in self._text(actual).casefold()
            elif operator == "in":
                if not isinstance(expected, list):
                    raise TabularError("in筛选的value必须是数组")
                matched = any(self._compare(actual, item) == 0 for item in expected)
            else:
                compared = self._compare(actual, expected)
                matched = {
                    "eq": compared == 0,
                    "ne": compared != 0,
                    "gt": compared > 0,
                    "gte": compared >= 0,
                    "lt": compared < 0,
                    "lte": compared <= 0,
                }[operator]
            if not matched:
                return False
        return True

    def _record(
        self, headers: list[str], values: list[Any]
    ) -> tuple[dict[str, Any], int]:
        record: dict[str, Any] = {}
        truncated = 0
        for index, header in enumerate(headers):
            value, clipped = self._cell(values[index] if index < len(values) else None)
            record[header] = value
            truncated += int(clipped)
        return record, truncated

    @staticmethod
    def _validate_columns(headers: list[str], columns: list[str]) -> None:
        missing = [column for column in columns if column not in headers]
        if missing:
            raise TabularError(f"列不存在：{', '.join(missing)}")

    def inspect(
        self, path_value: str, *, sheet: str | None = None, sample_rows: int = 50
    ) -> dict[str, Any]:
        path = self._path(path_value)
        sample_rows = min(200, max(1, int(sample_rows)))
        with self._source(path, sheet) as (meta, headers, rows):
            profiles = {
                header: {"nonempty": 0, "types": set(), "samples": []}
                for header in headers
            }
            scanned = 0
            has_more = False
            for row_number, values in rows:
                if scanned >= self.max_scan_rows:
                    has_more = True
                    break
                scanned += 1
                if scanned > sample_rows:
                    continue
                record, _ = self._record(headers, values)
                for header, value in record.items():
                    if self._is_empty(value):
                        continue
                    profile = profiles[header]
                    profile["nonempty"] += 1
                    profile["types"].add(
                        "number" if self._number(value) is not None else type(value).__name__
                    )
                    text = self._text(value)
                    if text not in profile["samples"] and len(profile["samples"]) < 3:
                        profile["samples"].append(text[:200])
            columns = [
                {
                    "name": header,
                    "sample_nonempty": profile["nonempty"],
                    "inferred_types": sorted(profile["types"]) or ["empty"],
                    "samples": profile["samples"],
                }
                for header, profile in profiles.items()
            ]
        return {
            **self.workspace._item(path),
            **meta,
            "columns": columns,
            "column_count": len(headers),
            "sampled_rows": min(scanned, sample_rows),
            "scanned_rows": scanned,
            "scan_truncated": has_more,
            "read_only": True,
            "safety_warning": "只读取数据；不执行宏、公式、外链或写回操作。类型是样本推断，重要结论应核对原始行。",
        }

    def read_rows(
        self,
        path_value: str,
        *,
        sheet: str | None = None,
        columns: list[str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        sort_by: str | None = None,
        sort_order: str = "asc",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        path = self._path(path_value)
        filters = list(filters or [])
        if len(filters) > 10:
            raise TabularError("单次筛选条件不能超过10个")
        offset = max(0, int(offset))
        limit = min(self.max_rows_per_call, max(1, int(limit)))
        sort_order = sort_order.lower()
        if sort_order not in {"asc", "desc"}:
            raise TabularError("sort_order必须是asc或desc")
        with self._source(path, sheet) as (meta, headers, rows):
            selected = list(columns or headers)
            self._validate_columns(headers, selected)
            if sort_by:
                self._validate_columns(headers, [sort_by])
            matched: list[dict[str, Any]] = []
            scanned = 0
            truncated_cells = 0
            has_more = False
            for row_number, values in rows:
                if scanned >= self.max_scan_rows:
                    has_more = True
                    break
                scanned += 1
                record, clipped = self._record(headers, values)
                truncated_cells += clipped
                if not self._matches(record, filters):
                    continue
                matched.append(
                    {
                        "row_number": row_number,
                        **{column: record[column] for column in selected},
                    }
                )
            if sort_by:
                def sort_key(item: dict[str, Any]) -> tuple[int, Any]:
                    value = item.get(sort_by)
                    number = self._number(value)
                    comparable: Any = number if number is not None else self._text(value).casefold()
                    return (0, comparable)

                populated = [item for item in matched if not self._is_empty(item.get(sort_by))]
                empty = [item for item in matched if self._is_empty(item.get(sort_by))]
                try:
                    populated.sort(key=sort_key, reverse=sort_order == "desc")
                except TypeError:
                    populated.sort(
                        key=lambda item: self._text(item.get(sort_by)).casefold(),
                        reverse=sort_order == "desc",
                    )
                matched = populated + empty
            returned = matched[offset : offset + limit]
        return {
            **self.workspace._item(path),
            **meta,
            "columns": selected,
            "filters": filters,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "offset": offset,
            "limit": limit,
            "scanned_rows": scanned,
            "matched_rows": len(matched),
            "returned_rows": len(returned),
            "rows": returned,
            "truncated": has_more or offset + len(returned) < len(matched),
            "truncated_cells": truncated_cells,
            "read_only": True,
        }

    def summarize(
        self,
        path_value: str,
        *,
        sheet: str | None = None,
        group_by: list[str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = self._path(path_value)
        group_by = list(group_by or [])
        filters = list(filters or [])
        metrics = list(metrics or [])
        if len(group_by) > 2:
            raise TabularError("group_by最多允许2列")
        if len(filters) > 10:
            raise TabularError("单次筛选条件不能超过10个")
        if not metrics or len(metrics) > 20:
            raise TabularError("metrics数量必须在1到20之间")
        with self._source(path, sheet) as (meta, headers, rows):
            self._validate_columns(headers, group_by)
            normalized_metrics: list[dict[str, str | None]] = []
            aliases: set[str] = set(group_by)
            for metric in metrics:
                operation = str(metric.get("operation") or "")
                column = str(metric.get("column") or "") or None
                if operation not in ALLOWED_AGGREGATIONS:
                    raise TabularError(f"不支持的汇总操作：{operation}")
                if operation != "count_rows" and not column:
                    raise TabularError(f"{operation}必须指定column")
                if column:
                    self._validate_columns(headers, [column])
                alias = str(metric.get("alias") or "").strip() or (
                    "row_count" if operation == "count_rows" else f"{operation}_{column}"
                )
                if not re.fullmatch(r"[\w\u4e00-\u9fff .-]{1,100}", alias):
                    raise TabularError("汇总别名包含不支持的字符")
                if alias in aliases:
                    raise TabularError(f"汇总别名重复：{alias}")
                aliases.add(alias)
                normalized_metrics.append(
                    {"operation": operation, "column": column, "alias": alias}
                )
            groups: dict[tuple[str, ...], dict[str, Any]] = {}
            scanned = 0
            matched_rows = 0
            has_more = False
            for _, values in rows:
                if scanned >= self.max_scan_rows:
                    has_more = True
                    break
                scanned += 1
                record, _ = self._record(headers, values)
                if not self._matches(record, filters):
                    continue
                matched_rows += 1
                key = tuple(self._text(record[column]) for column in group_by)
                if key not in groups:
                    if len(groups) >= self.max_groups:
                        raise TabularError(
                            f"分组数量超过上限{self.max_groups}；请增加筛选或减少分组列"
                        )
                    groups[key] = {
                        "row_count": 0,
                        "metrics": [
                            {"count": 0, "sum": Decimal(0), "values": set(), "min": None, "max": None}
                            for _ in normalized_metrics
                        ],
                    }
                state = groups[key]
                state["row_count"] += 1
                for index, metric in enumerate(normalized_metrics):
                    operation = metric["operation"]
                    if operation == "count_rows":
                        continue
                    value = record[str(metric["column"])]
                    metric_state = state["metrics"][index]
                    if self._is_empty(value):
                        continue
                    metric_state["count"] += 1
                    if operation in {"sum", "average"}:
                        number = self._number(value)
                        if number is not None:
                            metric_state["sum"] += number
                        else:
                            metric_state["count"] -= 1
                    elif operation == "distinct_count":
                        metric_state["values"].add(self._text(value))
                    elif operation in {"min", "max"}:
                        current = metric_state[operation]
                        if current is None or (
                            operation == "min" and self._compare(value, current) < 0
                        ) or (
                            operation == "max" and self._compare(value, current) > 0
                        ):
                            metric_state[operation] = value
            result_groups: list[dict[str, Any]] = []
            for key, state in groups.items():
                item: dict[str, Any] = {
                    column: key[index] for index, column in enumerate(group_by)
                }
                for index, metric in enumerate(normalized_metrics):
                    operation = str(metric["operation"])
                    metric_state = state["metrics"][index]
                    if operation == "count_rows":
                        value: Any = state["row_count"]
                    elif operation == "count_nonempty":
                        value = metric_state["count"]
                    elif operation == "distinct_count":
                        value = len(metric_state["values"])
                    elif operation == "sum":
                        value = (
                            float(metric_state["sum"])
                            if metric_state["count"]
                            else None
                        )
                    elif operation == "average":
                        value = (
                            float(metric_state["sum"] / metric_state["count"])
                            if metric_state["count"]
                            else None
                        )
                    else:
                        value = metric_state[operation]
                    item[str(metric["alias"])] = value
                result_groups.append(item)
            if not group_by and not result_groups:
                empty_item = {}
                for metric in normalized_metrics:
                    empty_item[str(metric["alias"])] = 0 if metric["operation"] in {"count_rows", "count_nonempty", "sum", "distinct_count"} else None
                result_groups.append(empty_item)
        return {
            **self.workspace._item(path),
            **meta,
            "group_by": group_by,
            "filters": filters,
            "metrics": normalized_metrics,
            "scanned_rows": scanned,
            "matched_rows": matched_rows,
            "group_count": len(result_groups),
            "groups": result_groups,
            "scan_truncated": has_more,
            "read_only": True,
            "calculation_warning": "汇总只覆盖已扫描行；非数字值不会进入sum/average。scan_truncated为true时不得视为全表结论。",
        }

    def inspect_tool(self, _: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.inspect(
            str(arguments.get("path") or ""),
            sheet=arguments.get("sheet"),
            sample_rows=arguments.get("sample_rows", 50),
        )

    def read_rows_tool(self, _: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.read_rows(
            str(arguments.get("path") or ""),
            sheet=arguments.get("sheet"),
            columns=arguments.get("columns"),
            filters=arguments.get("filters"),
            sort_by=arguments.get("sort_by"),
            sort_order=arguments.get("sort_order", "asc"),
            offset=arguments.get("offset", 0),
            limit=arguments.get("limit", 100),
        )

    def summarize_tool(self, _: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.summarize(
            str(arguments.get("path") or ""),
            sheet=arguments.get("sheet"),
            group_by=arguments.get("group_by"),
            filters=arguments.get("filters"),
            metrics=arguments.get("metrics") or [],
        )
