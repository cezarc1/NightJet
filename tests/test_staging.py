from pathlib import Path

from nightjet.staging import stage_kubetorch_source


def write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stage_kubetorch_source_copies_source_and_selected_bundles(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    destination = tmp_path / "stage" / "nightjet"

    write(source / "pyproject.toml", "[project]\nname = 'nightjet'\n")
    write(source / "uv.lock", "lock")
    write(source / "README.md", "readme")
    write(source / "src" / "nightjet" / "__init__.py", "")
    write(source / "src" / "nightjet" / "model.py", "model")
    write(source / "configs" / "candidate.yaml", "model: {}\n")
    write(source / "docs" / "campaign.md", "docs")
    write(source / ".venv" / "bin" / "python", "do-not-copy")
    write(source / ".pytest_cache" / "README.md", "do-not-copy")
    write(source / ".git" / "config", "do-not-copy")
    write(source / "outputs" / "old" / "checkpoint.pt", "do-not-copy")
    write(source / "src" / "nightjet" / "__pycache__" / "model.pyc", "do-not-copy")

    bundle = source / "data" / "bundles" / "hvi-darkir"
    write(bundle / "bundle_manifest.json", "{}")
    write(bundle / "input_luma.npy", "input")
    write(bundle / "target_luma.npy", "target")
    write(bundle / "splits.json", "{}")

    result = stage_kubetorch_source(source_dir=source, output_dir=destination, bundle_dirs=[bundle])

    assert result.output_dir == destination
    assert result.bundle_dirs == [destination / "data" / "bundles" / "hvi-darkir"]
    assert (destination / "pyproject.toml").exists()
    assert (destination / "src" / "nightjet" / "model.py").exists()
    assert (destination / "configs" / "candidate.yaml").exists()
    assert (destination / "data" / "bundles" / "hvi-darkir" / "input_luma.npy").exists()
    assert not (destination / ".venv").exists()
    assert not (destination / ".git").exists()
    assert not (destination / ".pytest_cache").exists()
    assert not (destination / "outputs").exists()
    assert not (destination / "src" / "nightjet" / "__pycache__").exists()
