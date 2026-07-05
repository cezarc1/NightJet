import numpy as np

from nightjet.splits import SplitSpec, build_frame_splits


def test_build_frame_splits_is_deterministic_and_non_overlapping() -> None:
    first = build_frame_splits(frame_count=20, spec=SplitSpec(train=0.6, val=0.2, test=0.2, seed=7))
    second = build_frame_splits(
        frame_count=20, spec=SplitSpec(train=0.6, val=0.2, test=0.2, seed=7)
    )

    assert first == second
    assert len(first.train) == 12
    assert len(first.val) == 4
    assert len(first.test) == 4
    all_indices = np.array(first.train + first.val + first.test)
    assert sorted(all_indices.tolist()) == list(range(20))
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)
