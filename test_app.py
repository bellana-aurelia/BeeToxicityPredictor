import importlib.util
import os
from pathlib import Path
import tempfile


def test_dataset_path_is_resolved_relative_to_app_directory():
    project_root = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory() as tmp_dir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            spec = importlib.util.spec_from_file_location("app_module", project_root / "app.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert Path(module.DATASET_PATH).is_file(), "Dataset path should resolve from the app directory"
        finally:
            os.chdir(old_cwd)
