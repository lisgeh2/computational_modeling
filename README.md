# Neural Category Lab

A small local browser app for exploring neural-network category learning with two visual-stimulus features: spatial frequency and orientation. It needs no account, API key, package installation, or network data at runtime.

## Run locally

Node.js is used for the test suite. To serve the app, use Python 3:

```powershell
npm run serve
```

Open <http://localhost:4173>. The app is static and can also be served by any local HTTP server.

Run the model tests with:

```powershell
npm test
```

## Explore

- Choose a one-dimensional rule, a two-dimensional diagonal rule, or the synthetic XOR extension.
- Change the hidden-layer size, learning rate, and initialization seed.
- Train continuously, pause, apply one update, reset, or repeat the same seed.
- Select a training or test point to inspect its prediction. Click elsewhere in the plot to probe a prediction without adding a label or changing the weights.
- Select a neuron to inspect its bias and activation. The hidden-unit intervention temporarily zeros the selected unit and can be restored without changing learned weights.
- Compare training and test loss and accuracy. Evaluation does not update the model.

Inputs are normalized from the displayed 0–100 scale. The model uses tanh hidden units, a sigmoid output, binary cross-entropy, and full-batch gradient descent. A network with no hidden units is a linear classifier.

## Data provenance

The public Dryad workbook and codebook could not be retrieved in the build environment. The app therefore uses deterministic local illustrative data generated from documented category rules; it does not contain participant responses or claim to reproduce human results. The XOR task is a synthetic classroom extension. The displayed feature values are mapping placeholders, not physical measurement units.

## Files

- `index.html`: interface and controls.
- `styles.css`: responsive layout and visual styling.
- `app.js`: application state, interaction, and canvas visualizations.
- `model.js`: deterministic datasets and neural-network implementation.
- `tests/model.test.js`: numerical, learning, reproducibility, and evaluation tests.
