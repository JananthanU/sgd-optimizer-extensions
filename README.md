# SGD Optimizer Extensions for Linear Regression

Five hands-on extensions to a gradient-descent linear regression pipeline, built on a custom computational graph framework (`cgnodes.py`) — no PyTorch or TensorFlow involved.

## Experiments

| # | Extension | Key idea |
|---|-----------|----------|
| 1 | **Input Standardisation** | Zero-mean / unit-variance normalisation enables a single shared learning rate |
| 2 | **Early Stopping** | Halt training when loss falls below a threshold for `patience` epochs |
| 3 | **Learning Rate Decay on Plateau** | Halve the LR instead of stopping — allows continued refinement |
| 4 | **Quadratic Model** | Extend the hypothesis with an x² term; ~10 % lower MSE than linear |
| 5 | **Adam Optimizer** | First + second moment estimates with bias correction; converges within 34 units of the analytical minimum |

## Dataset

Rent prices of apartments in Lausanne (living area in m² → rent in CHF).  
Place the CSV at `data/lausanne-appart.csv` before running the notebook.

## Structure

```
├── sgd_optimizer_extensions.ipynb   # main notebook
├── cgnodes.py                       # custom computational graph (ValueNode, MSELossNode, …)
└── data/
    └── lausanne-appart.csv
```

## Requirements

```
numpy
pandas
matplotlib
```

## Highlights

- **`MSELossNode`** — custom operator node with forward pass (½·(ŷ−y)²) and analytic backward pass (∂J/∂ŷ = ŷ−y)
- All experiments compare against the closed-form normal-equations solution as benchmark
- Adam reaches a half-MSE objective within 0.03 % of the analytical minimum after 252 epochs
