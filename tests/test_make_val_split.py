from make_val_split import stratified_split


def _labels(counts: dict[int, int]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for label, count in counts.items():
        for i in range(count):
            labels[f"val_{label}_{i}.jpg"] = label
    return labels


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    labels = _labels({0: 10, 1: 10, 2: 10})
    assert stratified_split(labels, seed=251) == stratified_split(labels, seed=251)


def test_split_covers_every_image_exactly_once() -> None:
    labels = _labels({0: 7, 1: 4})
    assignment = stratified_split(labels, seed=1)

    assert set(assignment) == set(labels)
    assert set(assignment.values()) <= {"dev", "test"}


def test_split_is_roughly_half_per_class() -> None:
    # Global shuffle-then-halve would let a small class land entirely on one
    # side; per-class splitting keeps every class within one image of 50/50.
    labels = _labels({0: 11, 1: 8})
    assignment = stratified_split(labels, seed=251)

    for label, count in ((0, 11), (1, 8)):
        dev = sum(1 for name, split in assignment.items()
                   if labels[name] == label and split == "dev")
        assert abs(dev - count / 2) <= 0.5


def test_a_single_image_class_is_not_dropped() -> None:
    labels = _labels({0: 1})
    assignment = stratified_split(labels, seed=251)

    assert assignment == {"val_0_0.jpg": "dev"}
