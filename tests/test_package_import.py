"""Regression tests for package imports and the engine module CLI."""

import subprocess
import sys


def test_package_import_does_not_preload_engine():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rag_eval; "
            "assert 'rag_eval.engine' not in sys.modules; "
            "from rag_eval import RAGEngine; "
            "from rag_eval.engine import RAGEngine as DirectEngine; "
            "assert RAGEngine is DirectEngine",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_engine_module_help_has_no_runpy_warning():
    result = subprocess.run(
        [sys.executable, "-m", "rag_eval.engine", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--documents" in result.stdout
    assert "--benchmark" in result.stdout
    assert "RuntimeWarning" not in result.stderr
