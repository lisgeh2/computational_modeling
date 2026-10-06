import test from 'node:test';
import assert from 'node:assert/strict';
import { NeuralNetwork, makeDataset, binaryCrossEntropy, sigmoid } from '../model.js';

function makeRows() {
  return [
    { x: [0, 0], y: 0 },
    { x: [100, 100], y: 1 },
    { x: [0, 100], y: 1 },
    { x: [100, 0], y: 1 }
  ];
}

test('sigmoid and cross-entropy are numerically sensible', () => {
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
  assert.ok(Math.abs(binaryCrossEntropy(0.5, 1) - Math.log(2)) < 1e-12);
  assert.ok(binaryCrossEntropy(0.1, 1) > 0);
});

test('linear classifier learns a separable task without changing during testing', () => {
  const network = new NeuralNetwork(0, 7);
  network.weights2[0][0] = 0.1;
  network.weights2[0][1] = 0.1;
  network.bias2[0] = -0.1;
  const train = [
    { x: [0, 0], y: 0 },
    { x: [100, 0], y: 1 },
    { x: [0, 100], y: 1 }
  ];
  const test = [{ x: [100, 100], y: 1 }];
  const before = JSON.stringify({ weights1: network.weights1, bias1: network.bias1, weights2: network.weights2, bias2: network.bias2 });
  network.step(train, 0.1);
  const afterTrain = JSON.stringify({ weights1: network.weights1, bias1: network.bias1, weights2: network.weights2, bias2: network.bias2 });
  const beforeTest = afterTrain;
  network.evaluate(test);
  const afterTest = JSON.stringify({ weights1: network.weights1, bias1: network.bias1, weights2: network.weights2, bias2: network.bias2 });
  assert.notEqual(before, afterTrain);
  assert.equal(afterTest, beforeTest);
  assert.ok(network.evaluate(train).accuracy >= 0.95 || network.evaluate(test).accuracy >= 0.5);
});

test('hidden-layer backpropagation matches a finite-difference gradient', () => {
  const rows = [{ x: [37, 68], y: 1 }, { x: [81, 22], y: 0 }];
  const network = new NeuralNetwork(2, 17);
  const delta = 1e-6;
  const plus = network.clone();
  const minus = network.clone();
  plus.weights1[0][0] += delta;
  minus.weights1[0][0] -= delta;
  const numerical = (plus.evaluate(rows).loss - minus.evaluate(rows).loss) / (2 * delta);

  const trained = network.clone();
  const originalWeight = trained.weights1[0][0];
  const learningRate = 1e-4;
  trained.step(rows, learningRate);
  const analytical = (originalWeight - trained.weights1[0][0]) / learningRate;
  assert.ok(Math.abs(numerical - analytical) < 1e-7);
});

test('zeroing a hidden unit changes inference without changing parameters', () => {
  const network = new NeuralNetwork(2, 17);
  const input = [37, 68];
  const before = JSON.stringify({ weights1: network.weights1, bias1: network.bias1, weights2: network.weights2, bias2: network.bias2 });
  const normal = network.forward(input);
  const intervened = network.forward(input, 0);
  const after = JSON.stringify({ weights1: network.weights1, bias1: network.bias1, weights2: network.weights2, bias2: network.bias2 });
  assert.equal(intervened.activated[0], 0);
  assert.notEqual(intervened.output[0], normal.output[0]);
  assert.equal(after, before);
});

test('a hidden network learns the synthetic XOR task', () => {
  const dataset = makeDataset('xor', 2026);
  const network = new NeuralNetwork(4, 2026);
  for (let update = 0; update < 500; update += 1) {
    network.step(dataset.train, 0.3);
  }
  const result = network.evaluate(dataset.test);
  assert.ok(result.accuracy > .95, `expected > 95% test accuracy, got ${result.accuracy}`);
  assert.equal(dataset.train.length, 4);
  assert.equal(dataset.test.length, 4);
  assert.ok(dataset.train.every(trainingRow => dataset.test.every(testRow =>
    trainingRow.x[0] !== testRow.x[0] || trainingRow.x[1] !== testRow.x[1]
  )));
});

test('seeded reset reproduces the initial state', () => {
  const first = new NeuralNetwork(4, 123);
  const second = new NeuralNetwork(4, 123);
  const firstState = JSON.stringify({ weights1: first.weights1, bias1: first.bias1, weights2: first.weights2, bias2: first.bias2 });
  const secondState = JSON.stringify({ weights1: second.weights1, bias1: second.bias1, weights2: second.weights2, bias2: second.bias2 });
  assert.equal(firstState, secondState);
});

test('dataset preserves train/test separation and fixed conditions', () => {
  const first = makeDataset('one-dimensional', 2026);
  const second = makeDataset('one-dimensional', 2026);
  assert.equal(first.train.length, 80);
  assert.equal(first.test.length, 40);
  assert.deepEqual(first.train, second.train);
  assert.deepEqual(first.test, second.test);
  assert.equal(first.source, 'illustrative-fallback');
  assert.ok(first.train.every(item => item.y === (item.x[0] < 50 ? 0 : 1)));
});
