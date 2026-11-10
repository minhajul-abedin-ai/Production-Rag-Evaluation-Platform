"""Optional, durable, tenant-scoped SQLite document storage.

Single-process demo only. The application builds a fresh search index on startup.
Database encryption, backups and multi-worker consistency require a real deployment design.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import stat

from .models import Document


class SQLiteDocumentStore:
    def __init__(self, filename: str) -> None:
        if not filename or filename == ":memory:":
            raise ValueError("RAG_STORAGE_PATH must be a disk file")
        path = Path(filename).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink():
            raise ValueError("SQLite storage path must not be a symbolic link")
        if os.name == "posix":
            # Fail closed rather than silently exposing private source documents.
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("SQLite file must be regular and owner-only (chmod 600)")
        self.path = path
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents(
                    tenant TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (tenant, doc_id)
                )
            """)

    def _connect(self):
        return sqlite3.connect(str(self.path), timeout=10)

    def load(self, tenant: str) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM documents WHERE tenant = ? ORDER BY doc_id",
                (tenant,),
            ).fetchall()
        # Invalid or tampered persisted data must fail instead of leaking to another tenant.
        result = [Document.model_validate_json(row[0]) for row in rows]
        return result

    def replace(self, tenant: str, documents: list[Document]) -> None:
        # DELETE and INSERT commit as one transaction; no partial ingestion is visible.
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE tenant = ?", (tenant,))
            conn.executemany(
                "INSERT INTO documents (tenant, doc_id, payload) VALUES (?, ?, ?)",
                [(tenant, doc.doc_id, doc.model_dump_json()) for doc in documents],
            )
