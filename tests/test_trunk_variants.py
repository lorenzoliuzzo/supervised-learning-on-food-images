import pytest
import torch
from trunk_variants import (
    BUDGET,
    BY_KEY,
    VARIANTS,
    ConcatPool,
    SpatialAttentionPool,
    build_head,
    trunk,
)


def test_every_variant_key_is_unique() -> None:
    keys = [v.key for v in VARIANTS]
    assert len(keys) == len(set(keys))
    assert set(BY_KEY) == set(keys)


@pytest.mark.parametrize("key", sorted(BY_KEY))
def test_every_variant_builds_and_stays_under_budget(key: str) -> None:
    # The no-MaxPool rows exist to document a cost, not as candidates, but they
    # are still built by the sweep, so they still have to fit.
    model = BY_KEY[key].build()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params < BUDGET


@pytest.mark.parametrize("head", ["gap", "gap+gmp", "attention"])
def test_each_head_produces_class_logits(head: str) -> None:
    model = trunk((64, 128, 256, 512), (2, 2, 2, 1), head=head, num_classes=251)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 176, 176))
    assert out.shape == (2, 251)


def test_unknown_head_is_rejected_rather_than_silently_ignored() -> None:
    with pytest.raises(ValueError, match="unknown head"):
        build_head("bilinear", 512, 251)


def test_concat_pool_stacks_average_and_max_along_channels() -> None:
    x = torch.randn(2, 8, 6, 6)
    pooled = ConcatPool()(x)

    assert pooled.shape == (2, 16, 1, 1)
    assert torch.allclose(pooled[:, :8, 0, 0], x.mean(dim=(2, 3)), atol=1e-6)
    assert torch.allclose(pooled[:, 8:, 0, 0], x.amax(dim=(2, 3)), atol=1e-6)


def test_concat_pool_separates_maps_that_global_average_pooling_confuses() -> None:
    # The reason to try this head at all: two maps with the same mean and very
    # different peaks are identical to GAP and distinguishable here.
    flat = torch.full((1, 4, 6, 6), 1.0)
    peaked = torch.zeros(1, 4, 6, 6)
    peaked[:, :, 0, 0] = 36.0

    assert torch.allclose(flat.mean(dim=(2, 3)), peaked.mean(dim=(2, 3)))
    assert not torch.allclose(ConcatPool()(flat), ConcatPool()(peaked))


def test_attention_pooling_weights_are_a_distribution_over_the_map() -> None:
    # Softmax over the spatial positions, not a per-cell sigmoid gate: the
    # weights must sum to one, which is what keeps this on the same scale as GAP.
    pool = SpatialAttentionPool(16)
    weights = pool.score(torch.randn(2, 16, 6, 6)).flatten(2).softmax(dim=-1)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 1), atol=1e-6)


def test_attention_pooling_reduces_to_global_average_when_scores_are_uniform() -> None:
    pool = SpatialAttentionPool(16)
    # A zeroed final conv makes every position score equally, so the softmax is
    # uniform -- at which point this is exactly GAP. That is the sane starting
    # point for the head, and it pins the arithmetic.
    torch.nn.init.zeros_(pool.score[-1].weight)
    torch.nn.init.zeros_(pool.score[-1].bias)
    pool.eval()

    x = torch.randn(2, 16, 6, 6)
    with torch.no_grad():
        pooled = pool(x)

    assert pooled.shape == (2, 16, 1, 1)
    assert torch.allclose(pooled[:, :, 0, 0], x.mean(dim=(2, 3)), atol=1e-5)


def test_narrow_variants_keep_baselines_depth_and_downsample_structure() -> None:
    # #28 measures width alone. The 5-stage variant already showed that changing
    # the downsample structure costs more than any width change buys, so a narrow
    # variant that also moved a downsample would answer the wrong question.
    baseline = BY_KEY["baseline"]
    for key in ("narrow-384", "narrow-256"):
        variant = BY_KEY[key]
        assert variant.blocks == baseline.blocks
        assert len(variant.widths) == len(baseline.widths)
        assert variant.pool == baseline.pool
        assert variant.head == baseline.head


def test_narrow_variants_are_actually_narrower_than_baseline() -> None:
    baseline = sum(p.numel() for p in BY_KEY["baseline"].build().parameters())
    for key in ("narrow-384", "narrow-256"):
        assert sum(p.numel() for p in BY_KEY[key].build().parameters()) < baseline


def test_head_variants_change_only_the_head() -> None:
    baseline = BY_KEY["baseline"]
    for key in ("head-gapgmp", "head-attn"):
        variant = BY_KEY[key]
        assert variant.widths == baseline.widths
        assert variant.blocks == baseline.blocks
        assert variant.pool == baseline.pool
        assert variant.head != baseline.head
