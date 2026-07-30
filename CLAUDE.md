# supervised-learning-on-food-images — agent instructions

A university supervised-learning exam project: a CNN classifying the 251 classes
of FoodX-251.

## The hard constraint

`FoodCNN` must stay **under 10,000,000 trainable parameters**. This is the exam's
grading criterion, not a preference. Any change to `model.py` must keep
`tests/test_model.py::test_parameter_budget` green.

## Layout

- `model.py` — `FoodCNN`, the from-scratch architecture. The only file the
  parameter budget applies to.
- `main.py` — training/eval entry point, adapted from the PyTorch ImageNet
  reference example. Keeps that script's argparse-and-`main_worker` shape on
  purpose; don't restructure it into something else.
- `resnet.py` — the unmodified PyTorch reference script, kept as a baseline for
  comparison. **Do not edit it.**
- `food101.py` — Food-101 side experiments.
- `report/` — Typst sources for the report. Don't edit unless the issue says to.
- `plans/` — dated roadmaps, `YYYY-MM-DD-slug.md`. Update the checkboxes and the
  `Status:` line of the active plan as work lands, rather than rewriting history.
  Numbers in a plan must come from a run or a benchmark we actually did.
- `tests/` — pytest.

## Conventions

- Python 3.11+. Type hints on every function signature, tests included.
- `pathlib` over `os.path`.
- Comment *why*, never *what*. No docstrings unless asked.
- Lint and format with `ruff`; leave `ruff check .` clean.
- Tests are `pytest`, no classes, plain `def test_*() -> None:` functions.

## Testing without the dataset

`food251/` is gitignored and absent in CI, so **no test may read it**. Test the
model on random tensors (`torch.randn`) and test data-loading logic against
`tmp_path` fixtures. A test that needs real images is the wrong test.

Run: `python -m pytest -q`
