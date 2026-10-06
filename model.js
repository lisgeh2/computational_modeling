const EPS = 1e-8;

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function sigmoid(value) {
  if (value >= 0) {
    const exp = Math.exp(-value);
    return 1 / (1 + exp);
  }
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

export function binaryCrossEntropy(probability, label) {
  const p = clamp(probability, EPS, 1 - EPS);
  return -(label * Math.log(p) + (1 - label) * Math.log(1 - p));
}

export function makeDataset(preset, seed = 2026) {
  const random = mulberry32(seed);
  const points = [];
  const labels = [];
  const raw = [];

  if (preset === 'one-dimensional') {
    const categories = [0, 1];
    for (let i = 0; i < 120; i += 1) {
      const frequency = Math.round((random() * 0.92 + 0.04) * 100);
      const orientation = Math.round(random() * 100);
      const category = frequency < 50 ? categories[0] : categories[1];
      points.push([frequency, orientation]);
      labels.push(category);
      raw.push({ frequency, orientation, category });
    }
  } else if (preset === 'two-dimensional') {
    for (let i = 0; i < 120; i += 1) {
      const frequency = Math.round(random() * 100);
      const orientation = Math.round(random() * 100);
      const category = frequency + orientation < 100 ? 0 : 1;
      points.push([frequency, orientation]);
      labels.push(category);
      raw.push({ frequency, orientation, category });
    }
  } else {
    const corners = [
      [0, 0, 0],
      [0, 100, 1],
      [100, 0, 1],
      [100, 100, 0]
    ];
    const heldOut = [
      [10, 10, 0],
      [10, 90, 1],
      [90, 10, 1],
      [90, 90, 0]
    ];
    corners.concat(heldOut).forEach(([frequency, orientation, category]) => {
      points.push([frequency, orientation]);
      labels.push(category);
      raw.push({ frequency, orientation, category });
    });
  }

  const examples = points.map((x, i) => ({ x, y: labels[i], raw: raw[i] }));
  const shuffled = preset === 'xor' ? examples : examples.sort(() => random() - 0.5);
  const trainCount = preset === 'xor' ? 4 : 80;
  const train = shuffled.slice(0, trainCount);
  const test = shuffled.slice(trainCount);

  return {
    preset,
    source: 'illustrative-fallback',
    description: preset === 'one-dimensional'
      ? 'Illustrative rule-based condition: category depends on spatial frequency only; orientation is irrelevant.'
      : preset === 'two-dimensional'
        ? 'Illustrative information-integration condition: category depends on both spatial frequency and orientation.'
        : 'Synthetic XOR extension: category A is low–low/high–high; category B is low–high/high–low.',
    featureNames: ['Spatial frequency (0–100)', 'Orientation (0–100)'],
    featureScale: 100,
    x: train.map(item => item.x).concat(test.map(item => item.x)),
    y: train.map(item => item.y).concat(test.map(item => item.y)),
    train: train.map(item => ({ x: item.x.slice(), y: item.y, raw: item.raw })),
    test: test.map(item => ({ x: item.x.slice(), y: item.y, raw: item.raw })),
    all: shuffled.map(item => ({ x: item.x.slice(), y: item.y, raw: item.raw }))
  };
}

export function splitDataset(dataset, trainCount = 80) {
  return {
    train: dataset.train,
    test: dataset.test
  };
}

export class NeuralNetwork {
  constructor(hiddenUnits, seed = 2026) {
    const random = mulberry32(seed);
    this.hiddenUnits = hiddenUnits;
    this.seed = seed;
    this.inputSize = 2;
    this.outputSize = 1;
    this.weights1 = Array.from({ length: hiddenUnits }, () =>
      Array.from({ length: this.inputSize }, () => random() * 0.4 - 0.2)
    );
    this.bias1 = Array.from({ length: hiddenUnits }, () => random() * 0.4 - 0.2);
    const outputWidth = hiddenUnits > 0 ? hiddenUnits : this.inputSize;
    this.weights2 = Array.from({ length: this.outputSize }, () =>
      Array.from({ length: outputWidth }, () => random() * 0.4 - 0.2)
    );
    this.bias2 = Array.from({ length: this.outputSize }, () => random() * 0.4 - 0.2);
  }

  clone() {
    const copy = new NeuralNetwork(this.hiddenUnits, this.seed);
    copy.weights1 = this.weights1.map(row => row.slice());
    copy.bias1 = this.bias1.slice();
    copy.weights2 = this.weights2.map(row => row.slice());
    copy.bias2 = this.bias2.slice();
    return copy;
  }

  forward(x, disabledHiddenIndex = null) {
    const normalized = x.map(value => value / 100);
    if (this.hiddenUnits === 0) {
      const linear = normalized[0] * this.weights2[0][0] + normalized[1] * this.weights2[0][1];
      const output = sigmoid(linear + this.bias2[0]);
      return { hidden: [], output: [output], activated: [normalized] };
    }

    const hiddenPre = this.weights1.map((row, unit) =>
      row.reduce((sum, weight, inputIndex) => sum + weight * normalized[inputIndex], 0) + this.bias1[unit]
    );
    const hidden = hiddenPre.map((value, index) =>
      index === disabledHiddenIndex ? 0 : Math.tanh(value)
    );
    const outputLinear = hidden.reduce((sum, value, index) => sum + value * this.weights2[0][index], 0) + this.bias2[0];
    const output = sigmoid(outputLinear);
    return { hidden: hiddenPre, activated: hidden, output: [output] };
  }

  evaluate(rows) {
    const predictions = rows.map(item => this.forward(item.x));
    const probabilities = predictions.map(result => result.output[0]);
    const totalLoss = rows.reduce((sum, item, index) => {
      const probability = probabilities[index];
      return sum + binaryCrossEntropy(probability, item.y);
    }, 0);
    const correct = rows.filter((item, index) => (probabilities[index] >= 0.5) === (item.y === 1)).length;
    const accuracy = rows.length ? correct / rows.length : 0;
    return {
      loss: totalLoss / rows.length,
      accuracy,
      probabilities,
      predictions,
      correct
    };
  }

  step(rows, learningRate) {
    if (!rows.length) return { loss: 0, accuracy: 0 };

    const weight1Gradient = Array.from({ length: this.hiddenUnits }, () => [0, 0]);
    const bias1Gradient = Array(this.hiddenUnits).fill(0);
    const weight2Gradient = Array.from({ length: 1 }, () => Array(this.hiddenUnits || this.inputSize).fill(0));
    const bias2Gradient = [0];
    const normalizedBatch = rows.map(row => row.x.map(value => value / 100));

    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const row = rows[rowIndex];
      const normalized = normalizedBatch[rowIndex];
      const forward = this.forward(row.x);
      const probability = forward.output[0];
      const outputError = probability - row.y;
      const outputGradient = outputError / rows.length;
      bias2Gradient[0] += outputGradient;

      if (this.hiddenUnits === 0) {
        weight2Gradient[0][0] += normalized[0] * outputGradient;
        weight2Gradient[0][1] += normalized[1] * outputGradient;
      } else {
        const hiddenGradients = forward.activated.map((hiddenValue, hiddenIndex) => {
          const hiddenError = this.weights2[0][hiddenIndex] * outputGradient;
          const hiddenGradient = (1 - hiddenValue * hiddenValue) * hiddenError;
          weight2Gradient[0][hiddenIndex] += hiddenValue * outputGradient;
          bias1Gradient[hiddenIndex] += hiddenGradient;
          weight1Gradient[hiddenIndex][0] += hiddenGradient * normalized[0];
          weight1Gradient[hiddenIndex][1] += hiddenGradient * normalized[1];
          return hiddenGradient;
        });
        void hiddenGradients;
      }
    }

    if (this.hiddenUnits === 0) {
      this.weights2[0][0] -= learningRate * weight2Gradient[0][0];
      this.weights2[0][1] -= learningRate * weight2Gradient[0][1];
    } else {
      for (let index = 0; index < this.hiddenUnits; index += 1) {
        this.bias1[index] -= learningRate * bias1Gradient[index];
        this.weights1[index][0] -= learningRate * weight1Gradient[index][0];
        this.weights1[index][1] -= learningRate * weight1Gradient[index][1];
        this.weights2[0][index] -= learningRate * weight2Gradient[0][index];
      }
    }
    this.bias2[0] -= learningRate * bias2Gradient[0];

    return this.evaluate(rows);
  }
}

export function finiteDifferenceCheck() {
  const network = new NeuralNetwork(0, 17);
  network.weights2[0][0] = 0.3;
  network.weights2[0][1] = -0.2;
  network.bias2[0] = 0.1;
  const rows = [{ x: [50, 25], y: 1 }];
  const base = network.evaluate(rows).loss;
  const delta = 1e-6;
  network.weights2[0][0] += delta;
  const plus = network.evaluate(rows).loss;
  network.weights2[0][0] -= 2 * delta;
  const minus = network.evaluate(rows).loss;
  const derivative = (plus - minus) / (2 * delta);
  return { base, derivative, analyticalGradient: outputGradientForTinyExample(network, rows[0]) };
}

function outputGradientForTinyExample(network, row) {
  const p = network.forward(row.x).output[0];
  return (p - row.y) * 0.01;
}

export function testClassification(network, rows) {
  return network.evaluate(rows);
}
