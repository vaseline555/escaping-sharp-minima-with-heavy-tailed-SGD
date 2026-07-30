# Theta — Truncated Heavy-Tailed SGD

Minimal wrapper that injects Lomax (Type-II Pareto) noise into any PyTorch
optimizer.

## Install

```bash
pip install -e .
```

## Usage

```python
from theta import Theta
from torch.optim import SGD

optimizer = Theta(
    model.parameters(), base=SGD, lr=0.1,
    alpha=1.4, scale=0.5, clip=1.0, tail_prob=0.95,
    momentum=0.9, weight_decay=1e-4,   # forwarded to SGD
)

for x, y, x_plus, y_plus, x_minus, y_minus in loader:
    noise = optimizer.sample_noise()
    optimizer.zero_grad()

    if noise.active:
        z = noise.value
        loss_minus = criterion(model(x_minus), y_minus)
        ((-z) * loss_minus).backward()

        loss_plus = criterion(model(x_plus), y_plus)
        (z * loss_plus).backward()

    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

### Parameters

| Parameter   | Description                                               |
| ----------- | --------------------------------------------------------- |
| `base`      | Optimizer class to wrap (e.g. `SGD`, `Adam`)              |
| `lr`        | Learning rate (forwarded to base)                         |
| `alpha`     | Lomax shape (tail index)                                  |
| `scale`     | Lomax scale                                               |
| `clip`      | Max gradient norm (`None` to disable)                     |
| `tail_prob` | CDF threshold for tail-gating (`0.0` = always inject)    |
