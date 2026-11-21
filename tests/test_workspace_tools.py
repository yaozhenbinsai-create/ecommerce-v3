from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.database import Database
from app.env_store import EnvStore
from app.main import create_app
from app.workspace_service import WorkspaceError, WorkspaceService
from app.tabular_service import TabularError, TabularService


class NoopCodex:
    def close(self):
        return None


def make_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": "Market Brief", "/Author": "Boss Agent Test"})
    with path.open("wb") as handle:
        writer.write(handle)


def test_workspace_text_listing_and_boundary(tmp_path: Path):
    root = tmp_path / "workspace"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "market.md").write_text(
        "标题\n市场规模增长\n结论", encoding="utf-8"
    )
    (root / "ignored.py").write_text("secret = True", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    service = WorkspaceService(root)

    listed = service.list_files(query="market")
    assert [item["path"] for item in listed["items"]] == ["nested/market.md"]
    assert "ignored.py" not in [item["path"] for item in service.list_files()["items"]]
    read = service.read_text("nested/market.md", start_line=2, max_lines=1)
    assert read["content"] == "市场规模增长"
    assert read["line_count"] == 3
    assert len(read["content_sha256"]) == 64
    with pytest.raises(WorkspaceError, match="workspace边界"):
        service.read_text("../outside.txt")
    with pytest.raises(WorkspaceError, match="相对路径"):
        service.read_text(str(outside))


def test_workspace_pdf_info_read_and_search(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    pdf_path = root / "market-brief.pdf"
    make_text_pdf(pdf_path, "Market growth reached 42 percent in 2026")
    service = WorkspaceService(root)

    info = service.pdf_info("market-brief.pdf")
    assert info["page_count"] == 1
    assert info["metadata"]["title"] == "Market Brief"
    pages = service.read_pdf("market-brief.pdf", start_page=1, page_count=5)
    assert pages["returned_pages"] == 1
    assert "42 percent" in pages["pages"][0]["text"]
    result = service.search_pdf("market-brief.pdf", "growth")
    assert result["match_count"] == 1
    assert result["matches"][0]["page"] == 1
    assert result["searched_pages"] == 1
    assert result["truncated"] is False


def test_workspace_api_settings_tools_and_errors(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'workspace.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("approved workspace notes", encoding="utf-8")
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
        workspace_path=root,
    )
    with TestClient(app) as client:
        tool_names = {item["name"] for item in client.get("/api/tools").json()}
        assert {
            "list_workspace_files",
            "read_workspace_text",
            "get_pdf_info",
            "read_pdf_pages",
            "search_pdf_text",
        } <= tool_names
        assert client.get("/api/workspace/files").json()["count"] == 1
        assert client.get(
            "/api/workspace/text", params={"path": "notes.txt"}
        ).json()["content"] == "approved workspace notes"
        assert client.get(
            "/api/workspace/text", params={"path": "../settings.yaml"}
        ).status_code == 400
        defaults = client.get("/api/settings/workspace").json()
        defaults["max_pdf_search_pages"] = 50
        saved = client.put("/api/settings/workspace", json=defaults)
        assert saved.status_code == 200
        assert app.state.workspace_service.max_pdf_search_pages == 50
        for name in (
            "list_workspace_files",
            "read_workspace_text",
            "get_pdf_info",
            "read_pdf_pages",
            "search_pdf_text",
        ):
            assert app.state.work_executor.default_tool_permissions[name] == "allow"


def test_tabular_csv_filter_sort_and_summary(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text(
        "region,product,revenue,users\n"
        "East,Alpha,120.5,10\n"
        "West,Beta,80,8\n"
        "East,Beta,200,12\n"
        "West,Alpha,150,15\n",
        encoding="utf-8",
    )
    workspace = WorkspaceService(root)
    service = TabularService(workspace)

    inspected = service.inspect("sales.csv", sample_rows=4)
    assert inspected["format"] == "csv"
    assert inspected["column_count"] == 4
    assert inspected["estimated_rows"] == 4
    assert {item["name"] for item in inspected["columns"]} == {
        "region",
        "product",
        "revenue",
        "users",
    }

    rows = service.read_rows(
        "sales.csv",
        columns=["region", "product", "revenue"],
        filters=[{"column": "revenue", "operator": "gte", "value": 100}],
        sort_by="revenue",
        sort_order="desc",
        limit=2,
    )
    assert rows["matched_rows"] == 3
    assert rows["returned_rows"] == 2
    assert [item["revenue"] for item in rows["rows"]] == ["200", "150"]
    assert rows["truncated"] is True

    summary = service.summarize(
        "sales.csv",
        group_by=["region"],
        metrics=[
            {"operation": "count_rows", "alias": "records"},
            {"operation": "sum", "column": "revenue", "alias": "revenue_sum"},
            {"operation": "average", "column": "users", "alias": "avg_users"},
            {
                "operation": "distinct_count",
                "column": "product",
                "alias": "products",
            },
        ],
    )
    groups = {item["region"]: item for item in summary["groups"]}
    assert groups["East"] == {
        "region": "East",
        "records": 2,
        "revenue_sum": 320.5,
        "avg_users": 11.0,
        "products": 2,
    }
    assert groups["West"]["revenue_sum"] == 230.0
    assert summary["scan_truncated"] is False


def test_tabular_xlsx_read_only_formula_and_safety(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Market"
    sheet.append(["company", "users", "calculated"])
    sheet.append(["Alpha", 60_000_000, "=B2/1000000"])
    sheet.append(["Beta", 45_000_000, "=B3/1000000"])
    notes = workbook.create_sheet("Notes")
    notes.append(["note"])
    notes.append(["source must be checked"])
    workbook.save(root / "market.xlsx")
    (root / "broken.xlsx").write_text("not a zip", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")

    workspace = WorkspaceService(root)
    service = TabularService(workspace)
    inspected = service.inspect("market.xlsx", sheet="Market")
    assert inspected["sheets"] == ["Market", "Notes"]
    assert inspected["formula_policy"].startswith("data_only")
    rows = service.read_rows(
        "market.xlsx",
        sheet="Market",
        filters=[{"column": "users", "operator": "gt", "value": 50_000_000}],
    )
    assert rows["matched_rows"] == 1
    assert rows["rows"][0]["company"] == "Alpha"
    assert rows["rows"][0]["calculated"] is None
    assert (root / "market.xlsx").exists()
    with pytest.raises(TabularError, match="工作表不存在"):
        service.inspect("market.xlsx", sheet="Missing")
    with pytest.raises(TabularError, match="结构无效"):
        service.inspect("broken.xlsx")
    with pytest.raises(TabularError, match="workspace边界"):
        service.inspect("../outside.csv")


def test_tabular_api_tools_permissions_and_settings(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'tabular.db'}")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "work_runtime:\n  auto_execute_enabled: false\n", encoding="utf-8"
    )
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "metrics.tsv").write_text(
        "team\tvalue\nA\t10\nB\t20\nA\t30\n", encoding="utf-8"
    )
    app = create_app(
        database=database,
        env_store=EnvStore(tmp_path / ".env"),
        codex=NoopCodex(),
        settings_path=settings_path,
        workspace_path=root,
    )
    with TestClient(app) as client:
        tools = {item["name"] for item in client.get("/api/tools").json()}
        assert {"inspect_table", "read_table_rows", "summarize_table"} <= tools
        listed = client.get("/api/workspace/files").json()
        assert listed["items"][0]["kind"] == "table"
        inspected = client.post(
            "/api/workspace/table/inspect", json={"path": "metrics.tsv"}
        )
        assert inspected.status_code == 200
        assert inspected.json()["delimiter"] == "\t"
        rows = client.post(
            "/api/workspace/table/rows",
            json={
                "path": "metrics.tsv",
                "filters": [{"column": "value", "operator": "gte", "value": 20}],
                "sort_by": "value",
                "sort_order": "desc",
            },
        )
        assert [item["value"] for item in rows.json()["rows"]] == ["30", "20"]
        summary = client.post(
            "/api/workspace/table/summary",
            json={
                "path": "metrics.tsv",
                "group_by": ["team"],
                "metrics": [
                    {"operation": "sum", "column": "value", "alias": "total"}
                ],
            },
        )
        assert summary.status_code == 200
        assert {item["team"]: item["total"] for item in summary.json()["groups"]} == {
            "A": 40.0,
            "B": 20.0,
        }
        defaults = client.get("/api/settings/workspace").json()
        defaults["max_table_scan_rows"] = 500
        saved = client.put("/api/settings/workspace", json=defaults)
        assert saved.status_code == 200
        assert app.state.tabular_service.max_scan_rows == 500
        for name in ("inspect_table", "read_table_rows", "summarize_table"):
            assert app.state.work_executor.default_tool_permissions[name] == "allow"
