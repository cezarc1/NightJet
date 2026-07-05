from pathlib import Path

from nightjet.candidates import load_candidate_configs


def test_candidate_configs_load_recommended_variants() -> None:
    candidates = load_candidate_configs(Path("configs/candidates"))
    names = {candidate.model.name for candidate in candidates}

    assert {
        "edge-v1-reco-s2-c16-f3",
        "edge-v1-reco-s2-c24-f3",
        "edge-v1-reco-s2-c16-f5",
    } <= names
