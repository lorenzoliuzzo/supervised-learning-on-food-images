import numpy as np
import pytest
import torch
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import Normalize

from main import eval_resolution, load_encoder_weights
from model import FoodCNN
from simsiam import (
    PredictionMLP,
    ProjectionMLP,
    SimSiamModel,
    TwoCropsTransform,
    UnlabeledImageDataset,
    build_eval_transform,
    build_ssl_transform,
    collapse_metrics,
    cosine_lr,
    extract_features,
    knn_accuracy,
    negative_cosine_similarity,
    param_groups,
    pretrain_dirs,
    simsiam_loss,
)


def test_forward_features_output_shape() -> None:
    model = FoodCNN(num_classes=251)
    model.eval()
    with torch.no_grad():
        out = model.forward_features(torch.randn(2, 3, 176, 176))
    assert out.shape == (2, 512)


def test_forward_still_produces_class_logits_after_the_refactor() -> None:
    model = FoodCNN(num_classes=251)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 176, 176))
    assert out.shape == (2, 251)


def test_projection_mlp_output_shape() -> None:
    projector = ProjectionMLP(in_dim=512, hidden_dim=64, out_dim=128)
    projector.eval()
    out = projector(torch.randn(4, 512))
    assert out.shape == (4, 128)


def test_prediction_mlp_output_shape() -> None:
    predictor = PredictionMLP(in_dim=128, hidden_dim=32)
    predictor.eval()
    out = predictor(torch.randn(4, 128))
    assert out.shape == (4, 128)


def test_negative_cosine_similarity_is_minus_one_for_identical_vectors() -> None:
    x = torch.randn(4, 16)
    loss = negative_cosine_similarity(x, x)
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)


def test_negative_cosine_similarity_does_not_backprop_into_z() -> None:
    p = torch.randn(4, 16, requires_grad=True)
    z = torch.randn(4, 16, requires_grad=True)
    negative_cosine_similarity(p, z).backward()
    assert p.grad is not None
    assert z.grad is None


def test_simsiam_loss_decreases_when_optimized_on_a_single_batch() -> None:
    torch.manual_seed(0)
    encoder = FoodCNN(num_classes=251)
    model = SimSiamModel(encoder, proj_dim=64, pred_hidden_dim=32)
    model.train()

    view1 = torch.randn(4, 3, 176, 176)
    view2 = torch.randn(4, 3, 176, 176)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    first_loss = None
    last_loss = None
    for _ in range(20):
        optimizer.zero_grad()
        p1, p2, z1, z2 = model(view1, view2)
        loss = simsiam_loss(p1, p2, z1, z2)
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert last_loss < first_loss


def test_two_crops_transform_produces_two_different_views() -> None:
    torch.manual_seed(0)
    image = Image.fromarray(np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8), mode='RGB')
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    two_crops = TwoCropsTransform(build_ssl_transform(normalize))

    view1, view2 = two_crops(image)

    assert view1.shape == (3, 176, 176)
    assert view2.shape == (3, 176, 176)
    assert not torch.equal(view1, view2)


def _write_jpgs(directory, count: int, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(array, mode='RGB').save(directory / f'{prefix}_{i:03d}.jpg')


def test_unlabeled_dataset_pools_images_from_every_directory(tmp_path) -> None:
    _write_jpgs(tmp_path / 'train_set', 3, 'train')
    _write_jpgs(tmp_path / 'test_set', 2, 'test')

    dataset = UnlabeledImageDataset([tmp_path / 'train_set', tmp_path / 'test_set'])

    assert len(dataset) == 5


def test_unlabeled_dataset_returns_views_without_a_label(tmp_path) -> None:
    _write_jpgs(tmp_path / 'train_set', 1, 'train')
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = UnlabeledImageDataset(
        [tmp_path / 'train_set'], TwoCropsTransform(build_ssl_transform(normalize)))
    view1, view2 = dataset[0]

    assert view1.shape == (3, 176, 176)
    assert view2.shape == (3, 176, 176)


def test_unlabeled_dataset_rejects_an_empty_pool(tmp_path) -> None:
    (tmp_path / 'train_set').mkdir()
    with pytest.raises(FileNotFoundError):
        UnlabeledImageDataset([tmp_path / 'train_set'])


def test_pretrain_dirs_adds_the_unlabeled_test_set(tmp_path) -> None:
    (tmp_path / 'train_set').mkdir()
    (tmp_path / 'test_set').mkdir()

    assert pretrain_dirs(tmp_path, 'train') == [tmp_path / 'train_set']
    assert pretrain_dirs(tmp_path, 'train+test') == [
        tmp_path / 'train_set', tmp_path / 'test_set']


def test_pretrain_dirs_explains_how_to_extract_a_missing_test_set(tmp_path) -> None:
    (tmp_path / 'train_set').mkdir()
    with pytest.raises(FileNotFoundError, match='unzip'):
        pretrain_dirs(tmp_path, 'train+test')


def test_collapse_metrics_separate_a_collapsed_representation_from_a_spread_one() -> None:
    torch.manual_seed(0)
    # A collapsed encoder maps every image to the same vector; a healthy one
    # spreads them over the sphere. The gate exists to tell these apart, so
    # both numbers must move in the right direction between the two.
    collapsed = torch.nn.functional.normalize(torch.ones(64, 512) + 1e-6, dim=1)
    spread = torch.nn.functional.normalize(torch.randn(64, 512), dim=1)

    collapsed_std, collapsed_rank = collapse_metrics(collapsed)
    spread_std, spread_rank = collapse_metrics(spread)

    assert collapsed_std < 1e-3
    assert spread_std > collapsed_std
    assert collapsed_rank < spread_rank


def test_knn_accuracy_is_perfect_when_queries_match_their_own_class(tmp_path) -> None:
    # Two well-separated clusters, one per class, with queries sitting on top
    # of the bank vectors -- anything but 100% here means the voting is wrong.
    bank = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    bank = torch.nn.functional.normalize(bank, dim=1)
    bank_labels = torch.tensor([0, 0, 1, 1])

    accuracy = knn_accuracy(
        bank, bank_labels, bank.clone(), bank_labels.clone(),
        k=2, temperature=0.07, num_classes=2)

    assert accuracy == pytest.approx(100.0)


def test_knn_accuracy_is_zero_when_every_label_is_wrong() -> None:
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    bank_labels = torch.tensor([0, 1])

    accuracy = knn_accuracy(
        bank, bank_labels, bank.clone(), torch.tensor([1, 0]),
        k=1, temperature=0.07, num_classes=2)

    assert accuracy == pytest.approx(0.0)


def test_load_encoder_weights_transfers_encoder_but_not_classifier(tmp_path) -> None:
    torch.manual_seed(0)
    pretrained_encoder = FoodCNN(num_classes=251)
    pretrained_model = SimSiamModel(pretrained_encoder, proj_dim=64, pred_hidden_dim=32)

    checkpoint_path = tmp_path / 'simsiam.pth.tar'
    torch.save({'epoch': 1, 'state_dict': pretrained_model.state_dict()}, checkpoint_path)

    fresh_model = FoodCNN(num_classes=251)
    load_encoder_weights(fresh_model, str(checkpoint_path))

    for key, value in pretrained_encoder.features.state_dict().items():
        assert torch.equal(fresh_model.features.state_dict()[key], value)

    assert not torch.equal(
        fresh_model.classifier[-1].weight, pretrained_model.encoder.classifier[-1].weight
    )


def test_load_encoder_weights_accepts_a_supervised_checkpoint(tmp_path) -> None:
    torch.manual_seed(0)
    supervised = FoodCNN(num_classes=251)
    checkpoint_path = tmp_path / 'supervised.pth.tar'
    # main.py's layout: `features.*` / `classifier.*`, no `encoder.` prefix.
    torch.save({'epoch': 86, 'state_dict': supervised.state_dict()}, checkpoint_path)

    fresh = FoodCNN(num_classes=251)
    load_encoder_weights(fresh, str(checkpoint_path))

    for key, value in supervised.features.state_dict().items():
        assert torch.equal(fresh.features.state_dict()[key], value)
    assert not torch.equal(
        fresh.classifier[-1].weight, supervised.classifier[-1].weight
    )


def test_load_encoder_weights_rejects_a_checkpoint_with_no_trunk(tmp_path) -> None:
    checkpoint_path = tmp_path / 'headless.pth.tar'
    torch.save({'state_dict': {'classifier.0.weight': torch.zeros(2, 2)}}, checkpoint_path)

    fresh = FoodCNN(num_classes=251)
    with pytest.raises(ValueError, match='no trunk weights'):
        load_encoder_weights(fresh, str(checkpoint_path))


def test_load_encoder_weights_rejects_a_partial_trunk(tmp_path) -> None:
    torch.manual_seed(0)
    supervised = FoodCNN(num_classes=251)
    state_dict = supervised.state_dict()
    # A trunk that is missing a tensor the current architecture expects, which
    # is what a checkpoint from a different FoodCNN generation looks like.
    dropped = next(k for k in state_dict if k.startswith('features.') and k.endswith('.weight'))
    del state_dict[dropped]
    checkpoint_path = tmp_path / 'partial.pth.tar'
    torch.save({'state_dict': state_dict}, checkpoint_path)

    fresh = FoodCNN(num_classes=251)
    with pytest.raises(ValueError, match='does not match this FoodCNN'):
        load_encoder_weights(fresh, str(checkpoint_path))


def test_param_groups_cover_every_parameter() -> None:
    model = SimSiamModel(FoodCNN(num_classes=251))
    grouped = {id(p) for group in param_groups(model) for p in group['params']}
    assert grouped == {id(p) for p in model.parameters()}


def test_predictor_lr_stays_constant_while_trunk_decays() -> None:
    model = SimSiamModel(FoodCNN(num_classes=251))
    optimizer = torch.optim.SGD(param_groups(model), lr=0.05, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=[
        lambda epoch: cosine_lr(epoch, 50),
        lambda _epoch: 1.0,
    ])

    trunk_lrs = []
    for _ in range(50):
        optimizer.step()
        trunk_lrs.append(optimizer.param_groups[0]['lr'])
        assert optimizer.param_groups[1]['lr'] == pytest.approx(0.05)
        scheduler.step()

    assert trunk_lrs[0] == pytest.approx(0.05)
    assert trunk_lrs[-1] < 1e-3
    assert trunk_lrs == sorted(trunk_lrs, reverse=True)


def _feature_loader(num_images: int = 6, batch_size: int = 2) -> torch.utils.data.DataLoader:
    images = torch.randn(num_images, 3, 176, 176)
    labels = torch.arange(num_images) % 251
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels), batch_size=batch_size)


def test_extract_features_returns_l2_normalized_rows() -> None:
    # knn_accuracy takes `chunk @ bank.T` as a cosine similarity, which is only
    # true if extract_features has already normalized both sides.
    features, _ = extract_features(
        _feature_loader(), FoodCNN(num_classes=251), torch.device('cpu'), torch.float32)

    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_extract_features_keeps_labels_aligned_with_the_loader() -> None:
    loader = _feature_loader(num_images=6, batch_size=2)

    features, labels = extract_features(
        loader, FoodCNN(num_classes=251), torch.device('cpu'), torch.float32)

    assert features.shape == (6, 512)
    assert torch.equal(labels, torch.arange(6) % 251)


def test_extract_features_restores_train_mode() -> None:
    # The probe runs between training epochs, so leaving the encoder in eval()
    # would silently freeze BatchNorm's running stats for the rest of the run.
    encoder = FoodCNN(num_classes=251)
    encoder.train()

    extract_features(_feature_loader(), encoder, torch.device('cpu'), torch.float32)

    assert encoder.training


def test_extract_features_builds_no_autograd_graph() -> None:
    # The bank is 25k x 512; retaining graph history over it would hold the
    # activations of every probe batch at once.
    features, _ = extract_features(
        _feature_loader(), FoodCNN(num_classes=251), torch.device('cpu'), torch.float32)

    assert not features.requires_grad


def test_build_eval_transform_evaluates_at_the_fixres_resolution() -> None:
    # Deliberately 224, not the 176 the SSL views are cropped to: FixRes
    # evaluates higher than it trains, and this pipeline has to match main.py's
    # validation exactly or a kNN number here is not comparable to the
    # supervised ones. eval_resolution(176) is (256, 224), the same pair.
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))

    tensor = build_eval_transform(normalize)(image)

    assert tensor.shape == (3, 224, 224)
    assert eval_resolution(176) == (256, 224)
