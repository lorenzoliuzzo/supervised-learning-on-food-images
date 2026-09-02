import pathlib
import sys

import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import v2

import main
from main import (
    WARMUP_EPOCHS,
    GeneralizedCrossEntropyLoss,
    SimilaritySmoothedCrossEntropyLoss,
    build_similarity_matrix,
    build_train_transform,
    checkpoint_weights,
    load_checkpoint_weights,
    select_amp_dtype,
    warmup_cosine_lr,
)
from model import FoodCNN


def test_select_amp_dtype_is_bf16_on_cpu() -> None:
    # No capability to check on CPU -- autocast there is only ever a no-op
    # (enabled=device.type == 'cuda' at every call site), so the dtype choice
    # is irrelevant; bf16 keeps this branch from calling the CUDA-only API.
    assert select_amp_dtype(torch.device("cpu")) is torch.bfloat16


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), torch.bfloat16),  # Ampere: bf16 Tensor Cores exist
        ((9, 0), torch.bfloat16),  # Hopper: still >= 8.0
        ((7, 5), torch.float16),  # Turing (e.g. Colab's Tesla T4): no bf16 Tensor Cores
        ((6, 1), torch.float16),  # Pascal: further below the bf16 cutoff
    ],
)
def test_select_amp_dtype_matches_compute_capability(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: torch.dtype
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert select_amp_dtype(torch.device("cuda")) is expected


def test_single_batch_overfit_gate() -> None:
    # The cheapest possible proof the trunk can learn at all -- the one thing
    # that would otherwise waste a 2-hour run discovering a wiring bug.
    torch.manual_seed(0)
    model = FoodCNN(num_classes=251)
    model.train()

    images = torch.randn(8, 3, 176, 176)
    target = torch.randint(0, 251, (8,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(images), target)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1


def test_channels_last_survives_the_forward_pass() -> None:
    model = FoodCNN(num_classes=251).to(memory_format=torch.channels_last)
    model.eval()
    x = torch.randn(2, 3, 176, 176).to(memory_format=torch.channels_last)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 251)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 1 / WARMUP_EPOCHS),
        (WARMUP_EPOCHS, 1.0),
        (89, pytest.approx(0.0, abs=1e-3)),
    ],
)
def test_warmup_cosine_lr_hits_expected_multipliers(epoch: int, expected: float) -> None:
    # 90-epoch run: linear warmup through epoch 4, peak at epoch 5 (cosine
    # progress 0), decayed to ~0 by the last epoch (cosine progress ~1).
    assert warmup_cosine_lr(epoch, epochs=90) == expected


def test_gce_loss_is_near_zero_for_a_confident_correct_prediction() -> None:
    criterion = GeneralizedCrossEntropyLoss(q=0.7)
    output = torch.zeros(1, 4)
    output[0, 2] = 20.0  # softmax ~= 1 on the true class

    assert criterion(output, torch.tensor([2])).item() < 1e-3


def test_gce_loss_is_bounded_unlike_cross_entropy() -> None:
    # The whole point of GCE: a confidently *wrong* prediction costs at most
    # 1/q, not the unbounded loss CE would assign it -- that cap is what
    # limits how much a single mislabeled example can dominate the gradient.
    criterion = GeneralizedCrossEntropyLoss(q=0.7)
    output = torch.zeros(1, 4)
    output[0, 0] = 20.0  # confidently predicts class 0

    loss = criterion(output, torch.tensor([2]))  # true class is 2
    assert loss.item() < 1 / 0.7 + 1e-3
    assert torch.isfinite(loss)


def test_similarity_matrix_survives_a_class_partnered_with_everything() -> None:
    # Leaves no non-partner classes to spread the remaining budget over, which
    # divided by zero. --similarity-pairs reads an arbitrary user CSV, so the
    # degenerate shape is reachable without editing code.
    matrix = build_similarity_matrix(3, [(0, 1), (0, 2)], smoothing=0.1, partner_frac=0.5)

    assert torch.allclose(matrix.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.isfinite(matrix).all()
    # Class 0 has no "other" bucket left, so its partners absorb the whole
    # wrong-class budget rather than half of it.
    assert matrix[0, 1] == matrix[0, 2]
    assert matrix[0, 1] > 0


def test_similarity_matrix_handles_a_single_class() -> None:
    matrix = build_similarity_matrix(1, [], smoothing=0.1)

    assert torch.allclose(matrix, torch.ones(1, 1), atol=1e-6)


def test_similarity_matrix_rows_sum_to_one() -> None:
    matrix = build_similarity_matrix(5, [(0, 1)], smoothing=0.1, partner_frac=0.5)

    assert torch.allclose(matrix.sum(dim=1), torch.ones(5), atol=1e-6)


def test_similarity_matrix_matches_uniform_smoothing_with_no_partners() -> None:
    # A class with no detected duplicate is exactly ordinary label smoothing
    # (nn.CrossEntropyLoss's convention: a uniform blend over *all* K
    # classes, so the true class also picks up a small eps/K bonus on top of
    # 1-eps) -- the whole point is this only changes behavior where evidence
    # exists.
    matrix = build_similarity_matrix(5, [], smoothing=0.1, partner_frac=0.5)

    expected_row = torch.full((5,), 0.1 / 5)
    expected_row[2] = 0.9 + 0.1 / 5
    assert torch.allclose(matrix[2], expected_row, atol=1e-6)


def test_similarity_matrix_gives_partners_more_mass_than_non_partners() -> None:
    # Class 0's only detected partner is class 1; classes 2-4 are unrelated
    # and should get less smoothing mass each than the partner does.
    matrix = build_similarity_matrix(5, [(0, 1)], smoothing=0.1, partner_frac=0.5)

    assert matrix[0, 1] > matrix[0, 2]
    assert matrix[0, 2] == matrix[0, 3] == matrix[0, 4]


def test_similarity_matrix_pairs_are_symmetric() -> None:
    # (0, 1) as a detected pair means each is the other's partner, not just
    # a one-directional relationship.
    matrix = build_similarity_matrix(5, [(0, 1)], smoothing=0.1, partner_frac=0.5)

    assert matrix[0, 1] == matrix[1, 0]


def test_similarity_smoothed_loss_matches_uniform_smoothing_with_no_partners() -> None:
    matrix = build_similarity_matrix(4, [], smoothing=0.1)
    criterion = SimilaritySmoothedCrossEntropyLoss(matrix)
    reference = nn.CrossEntropyLoss(label_smoothing=0.1)
    output = torch.randn(3, 4)
    target = torch.tensor([0, 2, 3])

    assert torch.allclose(criterion(output, target), reference(output, target), atol=1e-5)


def test_similarity_smoothed_loss_penalizes_confident_wrong_more_than_confident_correct() -> None:
    # Like ordinary label smoothing, this never reaches zero even for a
    # perfectly confident correct prediction -- the smoothing mass keeps a
    # loss floor that discourages overconfidence by design. What must still
    # hold is the ranking: confidently wrong costs much more than confidently
    # right.
    matrix = build_similarity_matrix(4, [(0, 1)], smoothing=0.1)
    criterion = SimilaritySmoothedCrossEntropyLoss(matrix)
    target = torch.tensor([2])

    output_correct = torch.zeros(1, 4)
    output_correct[0, 2] = 20.0  # confidently predicts the true class
    output_wrong = torch.zeros(1, 4)
    output_wrong[0, 0] = 20.0  # confidently predicts a wrong class

    assert criterion(output_correct, target).item() < criterion(output_wrong, target).item()


@pytest.mark.parametrize("mix", ["mixup", "cutmix"])
def test_mix_with_gce_is_refused_before_any_work_happens(
    monkeypatch: pytest.MonkeyPatch, mix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # GCE indexes the hard label, so this combination crashed on the first
    # batch -- after building the model and the loaders. argparse rejects it
    # up front instead; both guards in main() run before any device work.
    monkeypatch.setattr(sys, "argv", ["main.py", "--mix", mix, "--loss", "gce"])

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2
    assert "incompatible" in capsys.readouterr().err


def test_distributed_without_an_accelerator_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # main_worker only wraps the model in DistributedDataParallel under CUDA,
    # so this combination silently trained N unsynchronised CPU models.
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--multiprocessing-distributed", "--no-accel"])

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2
    assert "requires an accelerator" in capsys.readouterr().err


def test_evaluate_with_a_missing_checkpoint_refuses_to_report_a_number(
    tmp_path: pathlib.Path,
) -> None:
    # Previously printed "no checkpoint found" and then evaluated randomly
    # initialized weights, reporting ~0.4% top-1 as though it were a result.
    args = main.parser.parse_args([
        "--evaluate", "--no-accel", "--resume", str(tmp_path / "nope.pth.tar"),
    ])
    args.distributed = False

    with pytest.raises(SystemExit, match="nothing to evaluate"):
        main.main_worker(None, 1, args)


def test_similarity_smoothed_loss_accepts_the_soft_target_mixup_produces() -> None:
    # --mix mixup --loss sim used to die here on `smoothing_matrix[target]`:
    # Mixup hands down a float (N, K) distribution, not hard indices.
    matrix = build_similarity_matrix(4, [(0, 1)], smoothing=0.1)
    criterion = SimilaritySmoothedCrossEntropyLoss(matrix)
    output = torch.randn(3, 4)
    soft_target = torch.zeros(3, 4)
    soft_target[:, 0] = 0.7
    soft_target[:, 2] = 0.3

    loss = criterion(output, soft_target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_similarity_smoothed_loss_soft_target_agrees_with_the_hard_one_when_one_hot() -> None:
    # A one-hot soft target must take the identical path to the index lookup,
    # or the two --mix legs of an ablation aren't measuring the same loss.
    matrix = build_similarity_matrix(4, [(0, 1)], smoothing=0.1)
    criterion = SimilaritySmoothedCrossEntropyLoss(matrix)
    output = torch.randn(3, 4)
    hard_target = torch.tensor([0, 2, 3])
    one_hot = torch.nn.functional.one_hot(hard_target, num_classes=4).float()

    assert torch.allclose(criterion(output, one_hot), criterion(output, hard_target), atol=1e-6)


def test_similarity_smoothed_loss_soft_target_blends_the_matrix_rows() -> None:
    # A 50/50 mix of classes 0 and 2 should cost the mean of what each costs
    # alone -- cross-entropy is linear in the target, and that linearity is
    # what makes the matmul form correct rather than merely non-crashing.
    matrix = build_similarity_matrix(4, [(0, 1)], smoothing=0.1)
    criterion = SimilaritySmoothedCrossEntropyLoss(matrix)
    output = torch.randn(1, 4)
    blend = torch.zeros(1, 4)
    blend[0, 0] = blend[0, 2] = 0.5

    expected = 0.5 * (criterion(output, torch.tensor([0])) + criterion(output, torch.tensor([2])))
    assert torch.allclose(criterion(output, blend), expected, atol=1e-6)


def test_mixup_replaces_hard_targets_with_a_soft_label_over_the_batch() -> None:
    mixer = v2.MixUp(num_classes=251)
    images = torch.randn(4, 3, 8, 8)
    target = torch.randint(0, 251, (4,))

    _, mixed_target = mixer(images, target)

    assert mixed_target.shape == (4, 251)
    assert torch.allclose(mixed_target.sum(dim=1), torch.ones(4), atol=1e-5)


def test_cutmix_replaces_hard_targets_with_a_soft_label_over_the_batch() -> None:
    mixer = v2.CutMix(num_classes=251)
    images = torch.randn(4, 3, 8, 8)
    target = torch.randint(0, 251, (4,))

    _, mixed_target = mixer(images, target)

    assert mixed_target.shape == (4, 251)
    assert torch.allclose(mixed_target.sum(dim=1), torch.ones(4), atol=1e-5)


def _ema_pair(decay: float = 0.9) -> tuple[nn.Module, torch.optim.swa_utils.AveragedModel]:
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay)
    )
    # Two updates with a weight change in between, so the raw and smoothed
    # weights genuinely differ and a mix-up between them is detectable.
    ema_model.update_parameters(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema_model.update_parameters(model)
    return model, ema_model


def test_checkpoint_state_dict_holds_the_smoothed_weights_that_ship() -> None:
    # --evaluate, analyze_errors.py and significance_test.py all read
    # 'state_dict' and expect the weights that were actually validated.
    model, ema_model = _ema_pair()

    payload = checkpoint_weights(model, ema_model)

    assert torch.allclose(payload['state_dict']['weight'], ema_model.module.weight)
    assert torch.allclose(payload['raw_state_dict']['weight'], model.weight)
    assert not torch.allclose(payload['state_dict']['weight'], model.weight)


def test_checkpoint_without_ema_writes_only_the_raw_weights() -> None:
    model = nn.Linear(4, 4, bias=False)

    payload = checkpoint_weights(model, None)

    assert set(payload) == {'state_dict'}
    assert torch.allclose(payload['state_dict']['weight'], model.weight)


def test_ema_resume_restores_the_raw_trajectory_not_its_own_average() -> None:
    # The bug: --resume loaded the smoothed weights into `model` and paired
    # them with the raw run's optimizer momentum, so a resumed run continued
    # from its own moving average.
    model, ema_model = _ema_pair()
    payload = checkpoint_weights(model, ema_model)
    raw_before, ema_before = model.weight.clone(), ema_model.module.weight.clone()

    fresh_model = nn.Linear(4, 4, bias=False)
    fresh_ema = torch.optim.swa_utils.AveragedModel(
        fresh_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))
    load_checkpoint_weights(payload, fresh_model, fresh_ema)

    assert torch.allclose(fresh_model.weight, raw_before)
    assert torch.allclose(fresh_ema.module.weight, ema_before)


def test_ema_resume_restores_n_averaged_so_the_average_is_not_re_seeded() -> None:
    # AveragedModel seeds rather than blends while n_averaged == 0. Losing the
    # counter would make the first post-resume update discard the whole
    # average and jump to the raw weights.
    model, ema_model = _ema_pair()
    payload = checkpoint_weights(model, ema_model)

    fresh_model = nn.Linear(4, 4, bias=False)
    fresh_ema = torch.optim.swa_utils.AveragedModel(
        fresh_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))
    load_checkpoint_weights(payload, fresh_model, fresh_ema)

    assert int(fresh_ema.n_averaged) == int(ema_model.n_averaged) == 2

    # One more step must blend, not seed: the EMA stays nearer where it was
    # than the raw weights it is being pulled toward.
    with torch.no_grad():
        fresh_model.weight.add_(1.0)
    before = fresh_ema.module.weight.clone()
    fresh_ema.update_parameters(fresh_model)

    assert not torch.allclose(fresh_ema.module.weight, fresh_model.weight)
    assert ((fresh_ema.module.weight - before).abs().mean()
            < (fresh_ema.module.weight - fresh_model.weight).abs().mean())


def test_resuming_an_ema_run_from_a_non_ema_checkpoint_warns_and_seeds() -> None:
    # No 'raw_state_dict' means 'state_dict' is the raw trajectory and there
    # is no average to continue -- adding --ema to an in-flight run.
    model = nn.Linear(4, 4, bias=False)
    payload = checkpoint_weights(model, None)

    fresh_model = nn.Linear(4, 4, bias=False)
    fresh_ema = torch.optim.swa_utils.AveragedModel(
        fresh_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))

    with pytest.warns(UserWarning, match='no EMA state'):
        load_checkpoint_weights(payload, fresh_model, fresh_ema)

    assert torch.allclose(fresh_model.weight, model.weight)
    assert int(fresh_ema.n_averaged) == 0


def test_ema_model_moves_toward_new_weights_without_jumping_there() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9)
    )
    # The first update_parameters() call seeds the EMA at the current weights
    # rather than blending -- there is nothing to blend with yet. Only the
    # second call onward applies the decay, which is what train() relies on
    # over many SGD steps per epoch.
    ema_model.update_parameters(model)
    seeded = ema_model.module.weight.clone()

    with torch.no_grad():
        model.weight.add_(1.0)
    ema_model.update_parameters(model)

    # Decay 0.9 means the EMA should sit close to the seeded weights, not
    # jump straight to the new ones -- that's the smoothing EMA exists for.
    assert not torch.allclose(ema_model.module.weight, model.weight)
    assert not torch.allclose(ema_model.module.weight, seeded)
    distance_to_new = (ema_model.module.weight - model.weight).abs().mean()
    distance_to_old = (ema_model.module.weight - seeded).abs().mean()
    assert distance_to_old < distance_to_new


@pytest.mark.parametrize("augment", ["none", "trivial", "rand"])
def test_build_train_transform_survives_a_real_image(augment: str) -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = build_train_transform(augment, normalize)
    image = Image.new("RGB", (200, 150))

    out = transform(image)

    assert out.shape == (3, 176, 176)


def test_build_train_transform_none_has_no_augmentation_op() -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = build_train_transform("none", normalize)

    names = [type(t).__name__ for t in transform.transforms]
    assert "TrivialAugmentWide" not in names
    assert "RandAugment" not in names


def test_build_train_transform_defaults_to_the_imagenet_crop_scale() -> None:
    # Pins the inherited default. Every Phase C and Phase D number was measured
    # at 0.08, so changing it silently would retroactively change what those
    # runs mean.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop = build_train_transform("none", normalize).transforms[0]

    assert crop.scale == (0.08, 1.0)


def test_build_train_transform_crop_scale_min_reaches_the_crop() -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop = build_train_transform("rand", normalize, 0.4).transforms[0]

    assert crop.scale == (0.4, 1.0)


@pytest.mark.parametrize("crop_scale_min", [0.08, 0.25, 0.4])
def test_build_train_transform_output_shape_is_independent_of_crop_scale(
    crop_scale_min: float,
) -> None:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = build_train_transform("none", normalize, crop_scale_min)

    out = transform(Image.new("RGB", (200, 150)))

    assert out.shape == (3, 176, 176)
