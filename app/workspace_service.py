from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Any

from pypdf import PdfReader

from .tool_runtime import ToolContext, ToolRegistryError


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".xlsx"}


class WorkspaceError(ToolRegistryError):
    pass


class WorkspaceService:
    def __init__(
        self,
        root: Path,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.configure(settings or {})

    def configure(self, settings: dict[str, Any]) -> None:
        config = settings.get("workspace") or {}
        self.max_file_bytes = min(
            100_000_000,
            max(100_000, int(config.get("max_file_bytes", 20_000_000))),
        )
        self.max_text_chars = min(
            200_000,
            max(2_000, int(config.get("max_text_chars", 30_000))),
        )
        self.max_results = min(
            500, max(10, int(config.get("max_results", 100)))
        )
        self.max_pdf_pages_per_call = min(
            100, max(1, int(config.get("max_pdf_pages_per_call", 20)))
        )
        self.max_pdf_search_pages = min(
            1000, max(1, int(config.get("max_pdf_search_pages", 200)))
        )

    def _resolve(self, value: str, *, require_file: bool = False) -> Path:
        value = str(value or "").strip().replace("\\", "/")
        if not value or value.startswith(("/", "~")) or re.match(r"^[a-zA-Z]:", value):
            raise WorkspaceError("必须使用workspace内的相对路径")
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("路径超出workspace边界") from exc
        if candidate == self.root:
            raise WorkspaceError("请选择workspace内的具体文件")
        if require_file and (not candidate.exists() or not candidate.is_file()):
            raise WorkspaceError("文件不存在")
        return candidate

    def _validate_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise WorkspaceError(f"不支持的文件类型：{suffix or '无扩展名'}")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise WorkspaceError(
                f"文件超过读取上限：{size} > {self.max_file_bytes} bytes"
            )

    def _item(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        suffix = path.suffix.lower()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "kind": (
                "pdf"
                if suffix == ".pdf"
                else "table"
                if suffix in TABULAR_EXTENSIONS
                else "text"
            ),
        }

    def list_files(
        self,
        *,
        query: str = "",
        extension: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        query = query.strip().lower()
        extension = (extension or "").strip().lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if extension and extension not in SUPPORTED_EXTENSIONS:
            raise WorkspaceError("extension不是支持的文件类型")
        limit = min(self.max_results, max(1, int(limit or self.max_results)))
        items: list[dict[str, Any]] = []
        skipped = 0
        for current_root, directories, files in os.walk(self.root, followlinks=False):
            current_path = Path(current_root)
            safe_directories = []
            for name in directories:
                directory = current_path / name
                if directory.is_symlink():
                    skipped += 1
                    continue
                try:
                    directory.resolve().relative_to(self.root)
                except ValueError:
                    skipped += 1
                    continue
                safe_directories.append(name)
            directories[:] = safe_directories
            for name in files:
                path = current_path / name
                if path.is_symlink():
                    skipped += 1
                    continue
                if not path.is_file():
                    continue
                try:
                    path.resolve().relative_to(self.root)
                except ValueError:
                    skipped += 1
                    continue
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                relative = path.relative_to(self.root).as_posix()
                if extension and path.suffix.lower() != extension:
                    continue
                if query and query not in relative.lower():
                    continue
                if path.stat().st_size > self.max_file_bytes:
                    skipped += 1
                    continue
                items.append(self._item(path))
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break
        items.sort(key=lambda item: item["path"].lower())
        return {
            "workspace": self.root.name,
            "count": len(items),
            "items": items,
            "skipped": skipped,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        }

    def read_text(
        self,
        path_value: str,
        *,
        start_line: int = 1,
        max_lines: int = 500,
    ) -> dict[str, Any]:
        path = self._resolve(path_value, require_file=True)
        self._validate_file(path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise WorkspaceError("该文件不是支持的文本类型；PDF请使用PDF工具")
        start_line = max(1, int(start_line))
        max_lines = min(2000, max(1, int(max_lines)))
        raw = path.read_bytes()
        encoding = "utf-8"
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gb18030")
                encoding = "gb18030"
            except UnicodeDecodeError as exc:
                raise WorkspaceError("文本编码不是UTF-8或GB18030") from exc
        lines = text.splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        content = "\n".join(selected)
        truncated = len(content) > self.max_text_chars
        if truncated:
            content = content[: self.max_text_chars]
        return {
            **self._item(path),
            "encoding": encoding,
            "line_count": len(lines),
            "start_line": start_line,
            "returned_lines": len(selected),
            "content": content,
            "truncated": truncated or start_line - 1 + len(selected) < len(lines),
            "content_sha256": sha256(raw).hexdigest(),
        }

    def pdf_info(self, path_value: str) -> dict[str, Any]:
        path = self._resolve(path_value, require_file=True)
        self._validate_file(path)
        if path.suffix.lower() != ".pdf":
            raise WorkspaceError("该文件不是PDF")
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        raise WorkspaceError("PDF已加密，无法读取")
                except WorkspaceError:
                    raise
                except Exception as exc:
                    raise WorkspaceError("PDF已加密，无法读取") from exc
        except Exception as exc:
            if isinstance(exc, WorkspaceError):
                raise
            raise WorkspaceError(f"PDF无法解析：{exc}") from exc
        metadata = reader.metadata or {}
        return {
            **self._item(path),
            "page_count": len(reader.pages),
            "encrypted": bool(reader.is_encrypted),
            "metadata": {
                "title": getattr(metadata, "title", None),
                "author": getattr(metadata, "author", None),
                "subject": getattr(metadata, "subject", None),
                "creator": getattr(metadata, "creator", None),
                "producer": getattr(metadata, "producer", None),
                "creation_date": str(getattr(metadata, "creation_date", None) or "") or None,
            },
            "layout_warning": "文字提取不能证明表格、图表、图片或页面布局已正确理解；涉及视觉结构时需要页面渲染复核。",
        }

    def read_pdf(
        self,
        path_value: str,
        *,
        start_page: int = 1,
        page_count: int = 5,
    ) -> dict[str, Any]:
        path = self._resolve(path_value, require_file=True)
        self._validate_file(path)
        if path.suffix.lower() != ".pdf":
            raise WorkspaceError("该文件不是PDF")
        start_page = max(1, int(start_page))
        page_count = min(
            self.max_pdf_pages_per_call, max(1, int(page_count))
        )
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        raise WorkspaceError("PDF已加密，无法读取")
                except WorkspaceError:
                    raise
                except Exception as exc:
                    raise WorkspaceError("PDF已加密，无法读取") from exc
            total_pages = len(reader.pages)
            if start_page > total_pages:
                raise WorkspaceError("start_page超过PDF总页数")
            pages = []
            used_chars = 0
            for page_number in range(
                start_page, min(total_pages, start_page + page_count - 1) + 1
            ):
                text = reader.pages[page_number - 1].extract_text() or ""
                remaining = self.max_text_chars - used_chars
                clipped = text[: max(0, remaining)]
                pages.append(
                    {
                        "page": page_number,
                        "text": clipped,
                        "text_chars": len(text),
                        "truncated": len(clipped) < len(text),
                    }
                )
                used_chars += len(clipped)
                if used_chars >= self.max_text_chars:
                    break
        except WorkspaceError:
            raise
        except Exception as exc:
            raise WorkspaceError(f"PDF读取失败：{exc}") from exc
        return {
            **self._item(path),
            "total_pages": total_pages,
            "start_page": start_page,
            "returned_pages": len(pages),
            "pages": pages,
            "truncated": (
                not pages
                or pages[-1]["page"] < min(total_pages, start_page + page_count - 1)
                or any(page["truncated"] for page in pages)
            ),
            "layout_warning": "结果是分页文字层，不包含可靠的图表、图片、表格布局或扫描件OCR判断。",
        }

    def search_pdf(
        self,
        path_value: str,
        query: str,
        *,
        max_matches: int = 20,
    ) -> dict[str, Any]:
        path = self._resolve(path_value, require_file=True)
        self._validate_file(path)
        if path.suffix.lower() != ".pdf":
            raise WorkspaceError("该文件不是PDF")
        query = query.strip()
        if not query:
            raise WorkspaceError("query不能为空")
        max_matches = min(100, max(1, int(max_matches)))
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        raise WorkspaceError("PDF已加密，无法读取")
                except WorkspaceError:
                    raise
                except Exception as exc:
                    raise WorkspaceError("PDF已加密，无法读取") from exc
            matches = []
            needle = query.lower()
            page_limit = min(len(reader.pages), self.max_pdf_search_pages)
            for index, page in enumerate(reader.pages[:page_limit], start=1):
                text = page.extract_text() or ""
                lower = text.lower()
                position = lower.find(needle)
                if position < 0:
                    continue
                start = max(0, position - 300)
                end = min(len(text), position + len(query) + 700)
                matches.append(
                    {
                        "page": index,
                        "excerpt": re.sub(r"\s+", " ", text[start:end]).strip(),
                    }
                )
                if len(matches) >= max_matches:
                    break
        except Exception as exc:
            if isinstance(exc, WorkspaceError):
                raise
            raise WorkspaceError(f"PDF检索失败：{exc}") from exc
        return {
            **self._item(path),
            "query": query,
            "match_count": len(matches),
            "matches": matches,
            "searched_pages": page_limit,
            "total_pages": len(reader.pages),
            "truncated": page_limit < len(reader.pages) or len(matches) >= max_matches,
            "layout_warning": "检索仅覆盖PDF文字层；扫描图片和图表标签可能无法命中。",
        }

    def list_files_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.list_files(
            query=str(arguments.get("query") or ""),
            extension=arguments.get("extension"),
            limit=arguments.get("limit"),
        )

    def read_text_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.read_text(
            str(arguments.get("path") or ""),
            start_line=arguments.get("start_line", 1),
            max_lines=arguments.get("max_lines", 500),
        )

    def pdf_info_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.pdf_info(str(arguments.get("path") or ""))

    def read_pdf_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.read_pdf(
            str(arguments.get("path") or ""),
            start_page=arguments.get("start_page", 1),
            page_count=arguments.get("page_count", 5),
        )

    def search_pdf_tool(
        self, _: ToolContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.search_pdf(
            str(arguments.get("path") or ""),
            str(arguments.get("query") or ""),
            max_matches=arguments.get("max_matches", 20),
        )
