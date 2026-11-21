from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        connect_args = (
            {"check_same_thread": False, "timeout": 30}
            if url.startswith("sqlite")
            else {}
        )
        self.engine = create_engine(url, connect_args=connect_args)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def configure_sqlite_for_concurrency(dbapi_connection, _):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.engine.dialect.name != "sqlite":
            return
        inspector = inspect(self.engine)
        model_columns = {
            column["name"] for column in inspector.get_columns("model_entries")
        }
        model_additions = {
            "context_window": "ALTER TABLE model_entries ADD COLUMN context_window INTEGER",
            "max_output_tokens": "ALTER TABLE model_entries ADD COLUMN max_output_tokens INTEGER",
            "image_generation_endpoint": (
                "ALTER TABLE model_entries ADD COLUMN image_generation_endpoint VARCHAR(500)"
            ),
            "image_edit_endpoint": (
                "ALTER TABLE model_entries ADD COLUMN image_edit_endpoint VARCHAR(500)"
            ),
            "health_checked_at": (
                "ALTER TABLE model_entries ADD COLUMN health_checked_at DATETIME"
            ),
            "circuit_reset_at": (
                "ALTER TABLE model_entries ADD COLUMN circuit_reset_at DATETIME"
            ),
        }
        task_columns = {
            column["name"] for column in inspector.get_columns("pending_chat_tasks")
        }
        task_additions = {
            "checkpoint_json": "ALTER TABLE pending_chat_tasks ADD COLUMN checkpoint_json TEXT",
            "attempt_count": (
                "ALTER TABLE pending_chat_tasks ADD COLUMN attempt_count "
                "INTEGER NOT NULL DEFAULT 0"
            ),
            "task_kind": (
                "ALTER TABLE pending_chat_tasks ADD COLUMN task_kind "
                "VARCHAR(30) NOT NULL DEFAULT 'chat'"
            ),
            "parent_task_id": "ALTER TABLE pending_chat_tasks ADD COLUMN parent_task_id INTEGER",
            "result_message_id": "ALTER TABLE pending_chat_tasks ADD COLUMN result_message_id INTEGER",
            "finished_at": "ALTER TABLE pending_chat_tasks ADD COLUMN finished_at DATETIME",
        }
        chat_columns = {
            column["name"] for column in inspector.get_columns("chat_messages")
        }
        chat_additions = {
            "work_thread_id": "ALTER TABLE chat_messages ADD COLUMN work_thread_id INTEGER",
        }
        runtime_columns = {
            column["name"] for column in inspector.get_columns("agent_runtime_states")
        }
        runtime_additions = {
            "active_work_run_id": (
                "ALTER TABLE agent_runtime_states ADD COLUMN active_work_run_id INTEGER"
            ),
        }
        work_thread_columns = {
            column["name"] for column in inspector.get_columns("work_threads")
        } if inspector.has_table("work_threads") else set()
        work_thread_additions = {
            "project_id": (
                "ALTER TABLE work_threads ADD COLUMN project_id INTEGER "
                "REFERENCES work_projects(id)"
            ),
            "generation_number": (
                "ALTER TABLE work_threads ADD COLUMN generation_number INTEGER "
                "NOT NULL DEFAULT 1"
            ),
            "continuation_of_thread_id": (
                "ALTER TABLE work_threads ADD COLUMN continuation_of_thread_id INTEGER "
                "REFERENCES work_threads(id)"
            ),
        }
        tool_call_columns = {
            column["name"] for column in inspector.get_columns("tool_calls")
        } if inspector.has_table("tool_calls") else set()
        tool_call_additions = {
            "duration_ms": (
                "ALTER TABLE tool_calls ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
            ),
        }
        work_model_call_columns = {
            column["name"] for column in inspector.get_columns("work_model_calls")
        } if inspector.has_table("work_model_calls") else set()
        work_model_call_additions = {
            "prompt_hash": (
                "ALTER TABLE work_model_calls ADD COLUMN prompt_hash VARCHAR(64) "
                "NOT NULL DEFAULT ''"
            ),
            "prompt_chars": (
                "ALTER TABLE work_model_calls ADD COLUMN prompt_chars INTEGER "
                "NOT NULL DEFAULT 0"
            ),
        }
        evidence_columns = {
            column["name"] for column in inspector.get_columns("evidence")
        } if inspector.has_table("evidence") else set()
        evidence_additions = {
            "source_type": (
                "ALTER TABLE evidence ADD COLUMN source_type VARCHAR(30) "
                "NOT NULL DEFAULT 'unknown'"
            ),
            "source_publisher": "ALTER TABLE evidence ADD COLUMN source_publisher VARCHAR(300)",
            "claim_key": "ALTER TABLE evidence ADD COLUMN claim_key VARCHAR(200)",
            "verification_notes": "ALTER TABLE evidence ADD COLUMN verification_notes TEXT",
            "value_analysis_json": (
                "ALTER TABLE evidence ADD COLUMN value_analysis_json TEXT "
                "NOT NULL DEFAULT '{}'"
            ),
            "value_score": (
                "ALTER TABLE evidence ADD COLUMN value_score INTEGER NOT NULL DEFAULT 0"
            ),
            "review_status": (
                "ALTER TABLE evidence ADD COLUMN review_status VARCHAR(30) "
                "NOT NULL DEFAULT 'unreviewed'"
            ),
            "reviewer_agent_id": "ALTER TABLE evidence ADD COLUMN reviewer_agent_id INTEGER",
            "review_notes": "ALTER TABLE evidence ADD COLUMN review_notes TEXT",
            "reviewed_at": "ALTER TABLE evidence ADD COLUMN reviewed_at DATETIME",
        }
        research_candidate_columns = {
            column["name"] for column in inspector.get_columns("research_candidates")
        } if inspector.has_table("research_candidates") else set()
        research_candidate_additions = {
            "status": (
                "ALTER TABLE research_candidates ADD COLUMN status VARCHAR(30) "
                "NOT NULL DEFAULT 'queued'"
            ),
            "discovery_tool_call_id": (
                "ALTER TABLE research_candidates ADD COLUMN discovery_tool_call_id INTEGER"
            ),
            "read_tool_call_id": (
                "ALTER TABLE research_candidates ADD COLUMN read_tool_call_id INTEGER"
            ),
            "source_rank": "ALTER TABLE research_candidates ADD COLUMN source_rank INTEGER",
            "content_kind": (
                "ALTER TABLE research_candidates ADD COLUMN content_kind VARCHAR(40)"
            ),
            "source_text": "ALTER TABLE research_candidates ADD COLUMN source_text TEXT",
            "content_chars": (
                "ALTER TABLE research_candidates ADD COLUMN content_chars INTEGER "
                "NOT NULL DEFAULT 0"
            ),
            "disposition_reason": (
                "ALTER TABLE research_candidates ADD COLUMN disposition_reason TEXT"
            ),
            "updated_at": (
                "ALTER TABLE research_candidates ADD COLUMN updated_at DATETIME"
            ),
        }
        proposition_columns = {
            column["name"] for column in inspector.get_columns("expression_propositions")
        } if inspector.has_table("expression_propositions") else set()
        proposition_additions = {
            "theme_form": (
                "ALTER TABLE expression_propositions ADD COLUMN theme_form "
                "VARCHAR(100) NOT NULL DEFAULT '概念'"
            ),
            "creative_seeds_json": (
                "ALTER TABLE expression_propositions ADD COLUMN creative_seeds_json "
                "TEXT NOT NULL DEFAULT '[]'"
            ),
        }
        with self.engine.begin() as connection:
            if inspector.has_table("work_runs"):
                # Dependency waits are open runs but are never time-runnable.
                # Rebuild SQLite's partial uniqueness guard for existing DBs.
                connection.exec_driver_sql(
                    "DROP INDEX IF EXISTS uq_work_runs_open_thread"
                )
                connection.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_work_runs_open_thread "
                    "ON work_runs (work_thread_id) WHERE status IN "
                    "('queued','running','waiting_retry','waiting_schedule',"
                    "'waiting_dependency','waiting_user')"
                )
            for name, statement in model_additions.items():
                if name not in model_columns:
                    connection.exec_driver_sql(statement)
            connection.exec_driver_sql(
                "UPDATE model_entries SET protocol = 'openai_images' "
                "WHERE model_type = 'image' AND protocol = 'openai'"
            )
            connection.exec_driver_sql(
                "UPDATE model_entries SET protocol = 'openai_images', "
                "health_status = 'untested', "
                "health_message = '图片接口格式已迁移为 OpenAI Images API，请重新测试' "
                "WHERE model_type = 'image' AND protocol = 'openai_responses' "
                "AND LOWER(model_identifier) LIKE 'gpt-image-%'"
            )
            for name, statement in task_additions.items():
                if name not in task_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in chat_additions.items():
                if name not in chat_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in runtime_additions.items():
                if name not in runtime_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in proposition_additions.items():
                if name not in proposition_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in work_thread_additions.items():
                if name not in work_thread_columns:
                    connection.exec_driver_sql(statement)
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_work_threads_continuation_of_thread_id "
                "ON work_threads (continuation_of_thread_id)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_work_threads_project_id "
                "ON work_threads (project_id)"
            )
            connection.exec_driver_sql(
                "INSERT INTO work_projects ("
                "owner_agent_id,title,objective,success_criteria_json,constraints_json,"
                "status,current_root_thread_id,generation_count,stagnant_generations,"
                "progress_marker_json,summary,result_text,finished_at,created_at,updated_at"
                ") SELECT owner_agent_id,title,objective,success_criteria_json,constraints_json,"
                "CASE WHEN status='completed' THEN 'completed' "
                "WHEN status IN ('failed','cancelled') THEN 'blocked_external' "
                "ELSE 'running' END,id,1,0,'{}',summary,result_text,finished_at,created_at,updated_at "
                "FROM work_threads AS root WHERE root.parent_thread_id IS NULL "
                "AND root.continuation_of_thread_id IS NULL AND root.project_id IS NULL"
            )
            connection.exec_driver_sql(
                "UPDATE work_threads SET project_id=(SELECT id FROM work_projects "
                "WHERE current_root_thread_id=work_threads.id) "
                "WHERE parent_thread_id IS NULL AND continuation_of_thread_id IS NULL "
                "AND project_id IS NULL"
            )
            connection.exec_driver_sql(
                "WITH RECURSIVE lineage(id,project_id,generation_number) AS ("
                "SELECT id,project_id,generation_number FROM work_threads "
                "WHERE project_id IS NOT NULL "
                "UNION ALL "
                "SELECT child.id,lineage.project_id,lineage.generation_number "
                "FROM work_threads AS child JOIN lineage "
                "ON child.parent_thread_id=lineage.id WHERE child.project_id IS NULL "
                "UNION ALL "
                "SELECT successor.id,lineage.project_id,lineage.generation_number+1 "
                "FROM work_threads AS successor JOIN lineage "
                "ON successor.continuation_of_thread_id=lineage.id "
                "WHERE successor.project_id IS NULL"
                ") UPDATE work_threads SET "
                "project_id=(SELECT project_id FROM lineage WHERE lineage.id=work_threads.id LIMIT 1),"
                "generation_number=(SELECT generation_number FROM lineage "
                "WHERE lineage.id=work_threads.id LIMIT 1) "
                "WHERE id IN (SELECT id FROM lineage)"
            )
            connection.exec_driver_sql(
                "UPDATE work_projects SET generation_count=COALESCE(("
                "SELECT MAX(generation_number) FROM work_threads "
                "WHERE work_threads.project_id=work_projects.id),1),"
                "current_root_thread_id=COALESCE((SELECT id FROM work_threads "
                "WHERE work_threads.project_id=work_projects.id "
                "AND work_threads.parent_thread_id IS NULL "
                "ORDER BY generation_number DESC,id DESC LIMIT 1),current_root_thread_id)"
            )
            for name, statement in tool_call_additions.items():
                if name not in tool_call_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in work_model_call_additions.items():
                if name not in work_model_call_columns:
                    connection.exec_driver_sql(statement)
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_work_model_calls_prompt_hash "
                "ON work_model_calls (prompt_hash)"
            )
            for name, statement in evidence_additions.items():
                if name not in evidence_columns:
                    connection.exec_driver_sql(statement)
            for name, statement in research_candidate_additions.items():
                if name not in research_candidate_columns:
                    connection.exec_driver_sql(statement)
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_research_candidates_status "
                "ON research_candidates (status)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_research_candidates_discovery_tool_call_id "
                "ON research_candidates (discovery_tool_call_id)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_research_candidates_read_tool_call_id "
                "ON research_candidates (read_tool_call_id)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_evidence_value_score "
                "ON evidence (value_score)"
            )
