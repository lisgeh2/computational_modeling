import { NeuralNetwork, makeDataset, binaryCrossEntropy, sigmoid } from './model.js';

const $ = (selector) => document.querySelector(selector);
const canvas = $('#spaceCanvas');
const spaceContext = canvas.getContext('2d');
const activationCanvas = $('#activationCanvas');
const activationContext = activationCanvas.getContext('2d');
const learningCanvas = $('#learningCanvas');
const learningContext = learningCanvas.getContext('2d');

const state = {
  preset: 'one-dimensional',
  hiddenUnits: 4,
  learningRate: 0.5,
  seed: 2026,
  dataset: null,
  network: null,
  train: [],
  test: [],
  trainingHistory: [],
  testHistory: [],
  updates: 0,
  running: false,
  paused: false,
  selectedExample: null,
  selectedNeuron: null,
  selectedProbe: null,
  neuronDisabled: null,
  lastFrame: 0,
  animationTimer: null,
  initialNetwork: null
};

const presetMeta = {
  'one-dimensional': {
    title: 'One relevant dimension',
    label: 'Published rule-based condition',
    description: 'A fixed rule uses spatial frequency; orientation is irrelevant. This tests whether the network filters the irrelevant feature.',
    dataNote: 'Illustrative local fallback'
  },
  'two-dimensional': {
    title: 'Two relevant dimensions',
    label: 'Published information-integration condition',
    description: 'Category membership depends on both spatial frequency and orientation, forming a diagonal boundary.',
    dataNote: 'Illustrative local fallback'
  },
  xor: {
    title: 'XOR synthetic extension',
    label: 'Synthetic classroom extension',
    description: 'Category A is low–low/high–high and category B is low–high/high–low. A single linear boundary cannot represent this rule.',
    dataNote: 'Synthetic extension'
  }
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function forwardForDisplay(x) {
  return state.network.forward(x, state.neuronDisabled);
}

function resetModel({ preserveSeed = true } = {}) {
  const seed = preserveSeed ? state.seed : Number($('#seedInput').value);
  state.network = new NeuralNetwork(state.hiddenUnits, seed);
  state.initialNetwork = state.network.clone();
  state.updates = 0;
  state.trainingHistory = [];
  state.testHistory = [];
  state.running = false;
  state.selectedProbe = null;
  state.neuronDisabled = null;
  state.selectedNeuron = state.hiddenUnits > 0 ? { type: 'hidden', index: 0 } : { type: 'input', index: 0 };
  if (state.animationTimer) clearInterval(state.animationTimer);
  state.animationTimer = null;
  updateMetrics();
  renderAll();
}

function loadDataset() {
  const seed = Number($('#seedInput').value);
  state.seed = seed;
  state.dataset = makeDataset(state.preset, seed);
  state.train = state.dataset.train;
  state.test = state.dataset.test;
  state.selectedExample = state.train[Math.floor(state.train.length / 2)];
  state.selectedProbe = null;
  resetModel({ preserveSeed: true });
  state.trainingHistory = [{ update: 0, loss: null, accuracy: null, testLoss: null, testAccuracy: null }];
  state.testHistory = [];
  updateDatasetMeta();
  updateMetrics();
}

function updateDatasetMeta() {
  const meta = presetMeta[state.preset];
  $('#datasetTitle').textContent = meta.label;
  $('#datasetDescription').textContent = meta.description;
  $('#dataBadge').textContent = meta.dataNote;
  $('#dataSeedLabel').textContent = state.seed;
  $('#networkVisual').innerHTML = buildNetworkMarkup();
  bindNetworkEvents();
}

function buildNetworkMarkup() {
  const hidden = state.hiddenUnits > 0
    ? Array.from({ length: state.hiddenUnits }, (_, i) => `<button class="neuron hidden" data-neuron-type="hidden" data-neuron-index="${i}" title="Hidden neuron ${i + 1}">H${i + 1}</button>`).join('')
    : '<span class="neuron" style="grid-column:1/-1">No hidden layer</span>';
  const outputs = '<button class="neuron output" data-neuron-type="output" data-neuron-index="0">B</button>';
  const inputs = '<button class="neuron" data-neuron-type="input" data-neuron-index="0">SF</button><button class="neuron" data-neuron-type="input" data-neuron-index="1">Ori</button>';
  return `<div class="network-layer"><label>Input</label><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:.45rem">${inputs}</div></div><div class="arrows">→</div><div class="network-layer"><label>Hidden</label><div style="display:grid;grid-template-columns:repeat(${Math.min(state.hiddenUnits, 4)},1fr);gap:.45rem">${hidden}</div></div><div class="arrows">→</div><div class="network-layer"><label>Output</label>${outputs}</div>`;
}

function bindNetworkEvents() {
  document.querySelectorAll('[data-neuron-type]').forEach(button => {
    button.addEventListener('click', () => {
      state.selectedNeuron = { type: button.dataset.neuronType, index: Number(button.dataset.neuronIndex) };
      renderNetwork();
      renderInspector();
    });
  });
}

function neuronActivation(neuron) {
  if (neuron.type === 'input') return state.selectedExample?.x[neuron.index] ?? 0;
  if (neuron.type === 'output') return forwardForDisplay(state.selectedExample?.x ?? [50, 50]).output[0];
  if (neuron.type === 'hidden') {
    const result = forwardForDisplay(state.selectedExample?.x ?? [50, 50]);
    return state.neuronDisabled === neuron.index ? 0 : result.activated[neuron.index];
  }
  return 0;
}

function getNeuronDetails(neuron) {
  const x = state.selectedExample?.x ?? [50, 50];
  if (neuron.type === 'input') {
    const raw = x[neuron.index];
    return { bias: 0, weighted: raw / 100, activation: raw / 100, explanation: 'Raw normalized input value for the selected stimulus.' };
  }
  if (neuron.type === 'output') {
    const probability = forwardForDisplay(x).output[0];
    return { bias: state.network.bias2[0], weighted: probability, activation: probability, explanation: 'The output unit represents the sigmoid probability of category B.' };
  }
  const result = forwardForDisplay(x);
  const hiddenIndex = neuron.index;
  const weighted = state.network.weights1[hiddenIndex].reduce((sum, weight, inputIndex) => sum + weight * (x[inputIndex] / 100), 0) + state.network.bias1[hiddenIndex];
  const activation = state.neuronDisabled === hiddenIndex ? 0 : result.activated[hiddenIndex];
  return {
    bias: state.network.bias1[hiddenIndex],
    weighted,
    activation,
    explanation: `Hidden unit ${hiddenIndex + 1} combines the two inputs through tanh. Its activation is ${activation.toFixed(3)}.`
  };
}

function renderInspector() {
  const neuron = state.selectedNeuron;
  if (!neuron) return;
  const details = getNeuronDetails(neuron);
  let name = 'Input 1';
  if (neuron.type === 'input') name = neuron.index === 0 ? 'Spatial frequency' : 'Orientation';
  if (neuron.type === 'hidden') name = `Hidden neuron ${neuron.index + 1}`;
  if (neuron.type === 'output') name = 'Output: probability B';
  $('#neuronName').textContent = name;
  $('#biasValue').textContent = details.bias.toFixed(3);
  $('#weightedValue').textContent = details.weighted.toFixed(3);
  $('#activationValue').textContent = details.activation.toFixed(3);
  $('#neuronExplanation').textContent = details.explanation + (state.neuronDisabled === neuron.index ? ' Temporarily zeroed by intervention.' : '');
}

function renderNetwork() {
  $('#networkVisual').innerHTML = buildNetworkMarkup();
  document.querySelectorAll('[data-neuron-type]').forEach(button => {
    const type = button.dataset.neuronType;
    const index = Number(button.dataset.neuronIndex);
    button.classList.toggle('selected', state.selectedNeuron?.type === type && state.selectedNeuron?.index === index);
    button.classList.toggle('active', state.selectedNeuron?.type === type && state.selectedNeuron?.index === index);
  });
  bindNetworkEvents();
  renderInspector();
}

function drawHeatmap(canvas, context, values, xMin = 0, xMax = 100, yMin = 0, yMax = 100, label) {
  const width = canvas.width;
  const height = canvas.height;
  const margin = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#fff';
  context.fillRect(0, 0, width, height);
  const step = 2;
  for (let y = 0; y < 100; y += step) {
    for (let x = 0; x < 100; x += step) {
      const index = y * 50 + Math.floor(x / 2);
      const value = values[index] ?? 0;
      const color = value > .5 ? `rgba(${Math.round(85 + value * 95)}, ${Math.round(125 + value * 80)}, ${Math.round(255 - value * 100)}, .55)` : `rgba(112, 139, 255, ${.1 + value * .25})`;
      context.fillStyle = color;
      context.fillRect(margin.left + x / 100 * plotWidth, margin.top + (100 - y) / 100 * plotHeight, plotWidth / 50, plotHeight / 50);
    }
  }
  context.strokeStyle = '#d4deea';
  context.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const px = margin.left + i / 10 * plotWidth;
    context.beginPath(); context.moveTo(px, margin.top); context.lineTo(px, margin.top + plotHeight); context.stroke();
    const py = margin.top + i / 10 * plotHeight;
    context.beginPath(); context.moveTo(margin.left, py); context.lineTo(margin.left + plotWidth, py); context.stroke();
  }
  context.fillStyle = '#617083';
  context.font = '11px DM Sans'; context.textAlign = 'center';
  for (let i = 0; i <= 10; i += 1) {
    context.fillText(`${i * 10}`, margin.left + i / 10 * plotWidth, height - 13);
  }
  context.textAlign = 'right';
  for (let i = 0; i <= 10; i += 1) context.fillText(`${i * 10}`, margin.left - 7, margin.top + (10 - i) / 10 * plotHeight + 4);
  context.fillStyle = '#243447'; context.font = '600 11px DM Sans';
  context.textAlign = 'center'; context.fillText(label || 'Feature map', margin.left + plotWidth / 2, 13);
}

function pointToCanvas(point, width, height) {
  const margin = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  return {
    x: margin.left + point.x / 100 * plotWidth,
    y: margin.top + (100 - point.y) / 100 * plotHeight
  };
}

function renderSpace() {
  const width = canvas.width;
  const height = canvas.height;
  const margin = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  spaceContext.clearRect(0, 0, width, height);
  spaceContext.fillStyle = '#f8fafc'; spaceContext.fillRect(0, 0, width, height);
  const values = [];
  for (let y = 0; y < 100; y += 2) {
    for (let x = 0; x < 100; x += 2) {
      const point = [x, y];
      const p = forwardForDisplay(point).output[0];
      values.push(p);
      const neutral = [239, 243, 237];
      const categoryColor = p > .5 ? [217, 111, 54] : [20, 113, 91];
      const confidence = Math.pow(Math.abs(p - .5) * 2, .55);
      const color = neutral.map((channel, index) =>
        Math.round(channel + (categoryColor[index] - channel) * confidence)
      );
      spaceContext.fillStyle = `rgb(${color.join(',')})`;
      spaceContext.fillRect(margin.left + x / 100 * plotWidth, margin.top + (100 - y) / 100 * plotHeight, plotWidth / 50, plotHeight / 50);
    }
  }
  spaceContext.strokeStyle = '#d2dbe5'; spaceContext.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const px = margin.left + i / 10 * plotWidth;
    spaceContext.beginPath(); spaceContext.moveTo(px, margin.top); spaceContext.lineTo(px, margin.top + plotHeight); spaceContext.stroke();
    const py = margin.top + i / 10 * plotHeight;
    spaceContext.beginPath(); spaceContext.moveTo(margin.left, py); spaceContext.lineTo(margin.left + plotWidth, py); spaceContext.stroke();
  }
  const drawPoint = (point, label, color, radius = 6, marker = null) => {
    const p = pointToCanvas(point, width, height);
    spaceContext.beginPath(); spaceContext.arc(p.x, p.y, radius, 0, Math.PI * 2);
    spaceContext.fillStyle = color; spaceContext.fill();
    spaceContext.lineWidth = 2; spaceContext.strokeStyle = '#fff'; spaceContext.stroke();
    if (marker) { spaceContext.fillStyle = '#14253b'; spaceContext.font = '600 10px DM Sans'; spaceContext.fillText(marker, p.x + 10, p.y - 10); }
  };
  state.train.forEach(point => drawPoint({ x: point.x[0], y: point.x[1] }, 'Train', '#2769e8', 5, 'T'));
  state.test.forEach(point => drawPoint({ x: point.x[0], y: point.x[1] }, 'Test', '#f27c3d', 5, 'S'));
  if (state.selectedExample) drawPoint({ x: state.selectedExample.x[0], y: state.selectedExample.x[1] }, 'Selected', '#6c5ce7', 9, '★');
  if (state.selectedProbe) {
    drawPoint(state.selectedProbe, 'Probe', '#1d8f65', 8, 'P');
  }
  spaceContext.fillStyle = '#243447'; spaceContext.font = '600 11px DM Sans'; spaceContext.textAlign = 'center';
  spaceContext.fillText('Spatial frequency', margin.left + plotWidth / 2, height - 10);
  spaceContext.save(); spaceContext.translate(12, margin.top + plotHeight / 2); spaceContext.rotate(-Math.PI / 2); spaceContext.fillText('Orientation', 0, 0); spaceContext.restore();
  for (let i = 0; i <= 10; i += 1) {
    spaceContext.fillStyle = '#667585'; spaceContext.textAlign = 'center'; spaceContext.fillText(`${i * 10}`, margin.left + i / 10 * plotWidth, height - 17);
    spaceContext.textAlign = 'right'; spaceContext.fillText(`${i * 10}`, margin.left - 6, margin.top + (10 - i) / 10 * plotHeight + 4);
  }
  const gradient = spaceContext.createLinearGradient(margin.left, 0, margin.left + plotWidth, 0);
  gradient.addColorStop(0, 'rgba(95,117,255,.35)'); gradient.addColorStop(.5, 'rgba(110,92,231,.4)'); gradient.addColorStop(1, 'rgba(255,164,102,.45)');
  spaceContext.fillStyle = gradient; spaceContext.fillRect(margin.left, margin.top + plotHeight + 3, plotWidth, 4);
  spaceContext.fillStyle = '#657487'; spaceContext.font = '500 10px DM Sans'; spaceContext.textAlign = 'right'; spaceContext.fillText('p(B)', margin.left + plotWidth, margin.top + plotHeight + 16);
  void values;
}

function renderActivationMap() {
  if (state.hiddenUnits === 0) {
    activationContext.fillStyle = '#f1f4f8'; activationContext.fillRect(0,0,activationCanvas.width,activationCanvas.height);
    activationContext.fillStyle = '#667585'; activationContext.font = '600 13px DM Sans'; activationContext.textAlign = 'center'; activationContext.fillText('No hidden layer in this architecture', activationCanvas.width / 2, activationCanvas.height / 2);
    return;
  }
  const hiddenIndex = state.selectedNeuron?.type === 'hidden' ? state.selectedNeuron.index : 0;
  const activations = [];
  for (let y = 0; y <= 100; y += 5) {
    for (let x = 0; x <= 100; x += 5) {
      const result = forwardForDisplay([x, y]);
      const activation = state.neuronDisabled === hiddenIndex ? 0 : result.activated[hiddenIndex];
      activations.push(activation);
    }
  }
  const gradient = activationContext.createLinearGradient(0, 0, activationCanvas.width, 0);
  gradient.addColorStop(0, '#5b87ff'); gradient.addColorStop(.5, '#d7e5ff'); gradient.addColorStop(1, '#ffae63');
  activationContext.clearRect(0,0,activationCanvas.width,activationCanvas.height);
  activationContext.fillStyle = gradient;
  for (let i = 0; i < activations.length; i += 1) {
    const x = i % 21;
    const y = Math.floor(i / 21);
    activationContext.globalAlpha = .5 + activations[i] * .4;
    activationContext.fillRect(x * 28, y * 28, 28, 28);
  }
  activationContext.globalAlpha = 1;
  activationContext.strokeStyle = '#d5dce5'; activationContext.lineWidth = 1;
  for (let i = 0; i <= 20; i += 1) {
    activationContext.beginPath(); activationContext.moveTo(i*28,0); activationContext.lineTo(i*28,140); activationContext.stroke();
    activationContext.beginPath(); activationContext.moveTo(0,i*28); activationContext.lineTo(140,i*28); activationContext.stroke();
  }
  $('#mapLabel').textContent = `Hidden unit ${hiddenIndex + 1}`;
}

function renderLearning() {
  const width = learningCanvas.width;
  const height = learningCanvas.height;
  const margin = { top: 25, right: 25, bottom: 38, left: 54 };
  learningContext.clearRect(0,0,width,height);
  learningContext.fillStyle = '#fff'; learningContext.fillRect(0,0,width,height);
  const all = [...state.trainingHistory, ...state.testHistory];
  const maxUpdates = Math.max(1, all.length ? all[all.length - 1].update : 0);
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xPixel = (update) => margin.left + update / maxUpdates * plotW;
  const yPixel = (value) => margin.top + (1 - value) * plotH;
  learningContext.strokeStyle='#dce4ec';
  for (let i = 0; i <= 4; i += 1) { const y=margin.top+i/4*plotH; learningContext.beginPath(); learningContext.moveTo(margin.left,y); learningContext.lineTo(margin.left+plotW,y); learningContext.stroke(); }
  for (let i=0;i<=5;i++) { const x=margin.left+i/5*plotW; learningContext.beginPath(); learningContext.moveTo(x,margin.top); learningContext.lineTo(x,margin.top+plotH); learningContext.stroke(); }
  const drawSeries = (series, color, label) => {
    learningContext.beginPath();
    series.forEach((point, i) => { if (i === 0) learningContext.moveTo(xPixel(point.update), yPixel(point.loss)); else learningContext.lineTo(xPixel(point.update), yPixel(point.loss)); });
    learningContext.strokeStyle = color; learningContext.lineWidth = 2.5; learningContext.stroke();
    series.forEach(point => { learningContext.beginPath(); learningContext.arc(xPixel(point.update), yPixel(point.loss), 2.5,0,Math.PI*2); learningContext.fillStyle=color; learningContext.fill(); });
    learningContext.fillStyle='#567084'; learningContext.font='500 10px DM Sans'; learningContext.fillText(label, margin.left+plotW-65, margin.top+14);
  };
  const trainSeries = state.trainingHistory.filter(item=>item.loss !== null);
  if (trainSeries.length) drawSeries(trainSeries, '#2769e8', 'Training loss');
  if (state.testHistory.length) { learningContext.strokeStyle='#f27c3d'; learningContext.lineWidth=2; learningContext.stroke(); }
  const testSeries = state.testHistory.filter(item=>item.loss !== null);
  if (testSeries.length) drawSeries(testSeries, '#f27c3d', 'Testing loss');
  learningContext.fillStyle='#667585'; learningContext.font='11px DM Sans'; learningContext.textAlign='center';
  for(let i=0;i<=5;i++) learningContext.fillText(String(Math.round(i/5*maxUpdates)),margin.left+i/5*plotW,height-14);
  learningContext.textAlign='right'; learningContext.fillText('1',margin.left-8,margin.top+4); learningContext.fillText('0',margin.left-8,margin.top+plotH+4);
  $('#stepBadge').textContent = `${state.updates} updates`;
}

function updateMetrics() {
  const train = state.network ? state.network.evaluate(state.train) : { loss: 0, accuracy: 0 };
  const test = state.network ? state.network.evaluate(state.test) : { loss: 0, accuracy: 0 };
  $('#trainLoss').textContent = train.loss.toFixed(3);
  $('#testLoss').textContent = test.loss.toFixed(3);
  $('#trainAccuracy').textContent = `${Math.round(train.accuracy * 100)}%`;
  $('#testAccuracy').textContent = `${Math.round(test.accuracy * 100)}%`;
  if (!state.selectedExample) {
    $('#outputValue').textContent = 'Probability B = —';
  } else {
    const probability = forwardForDisplay(state.selectedExample.x).output[0].toFixed(3);
    $('#outputValue').textContent = state.selectedProbe
      ? `p(B) = ${probability} · unlabeled`
      : `p(B) = ${probability} · category ${state.selectedExample.y === 1 ? 'B' : 'A'}`;
  }
  $('#statusPill').classList.toggle('running', state.running);
  $('#statusPill').lastChild.textContent = state.running ? ' Training' : state.updates ? ' Paused' : ' Ready';
  $('#networkLiveLabel').textContent = state.running ? 'Updating' : 'Live state';
  renderLearning();
}

function updateSelectedInsight() {
  if (!state.selectedExample) return;
  const point = state.selectedExample;
  const p = forwardForDisplay(point.x).output[0];
  const item = state.selectedExample;
  const isProbe = Boolean(state.selectedProbe);
  const category = item.y === 1 ? 'Category B' : 'Category A';
  const sourceText = isProbe ? 'Unlabeled probe' : `${state.test.includes(item) ? 'Testing' : 'Training'} example`;
  $('#selectedTitle').textContent = isProbe ? sourceText : `${sourceText} · ${category}`;
  $('#selectedDescription').textContent = `Frequency ${item.x[0].toFixed(0)}, orientation ${item.x[1].toFixed(0)} · ${state.dataset?.featureNames?.[0] ?? 'Feature 1'} / ${state.dataset?.featureNames?.[1] ?? 'Feature 2'}`;
  $('#probabilityMetric').textContent = p.toFixed(3);
  $('#lossMetric').textContent = isProbe ? '—' : binaryCrossEntropy(p, item.y).toFixed(3);
  $('#stimulusPreview').innerHTML = `<span style="color:${p>.5?'#6c5ce7':'#2769e8'}">${makePreview(item.x)}</span>`;
  $('#outputValue').textContent = isProbe
    ? `p(B) = ${p.toFixed(3)} · unlabeled`
    : `p(B) = ${p.toFixed(3)} · category ${item.y === 1 ? 'B' : 'A'}`;
}

function makePreview(values) {
  const frequency = values[0] / 100;
  const orientation = values[1] / 100;
  const count = Math.round(3 + frequency * 10);
  const stripes = Array.from({ length: count }, () => '<i style="display:inline-block;width:3px;height:16px;margin:1px;background:#263c57;transform:rotate(' + Math.round(orientation * 45) + 'deg)"></i>').join('');
  return `<span style="font-size:${14 + frequency * 15}px">${stripes}</span>`;
}

function withSelectedExample(item) {
  state.selectedExample = item;
  state.selectedProbe = null;
  state.selectedNeuron = state.hiddenUnits > 0 ? { type: 'hidden', index: 0 } : { type: 'input', index: 0 };
  updateSelectedInsight();
  renderSpace();
  renderNetwork();
  renderActivationMap();
}

function probeAtCanvas(x, y) {
  const margin = { top: 20, right: 20, bottom: 40, left: 45 };
  const plotWidth = canvas.width - margin.left - margin.right;
  const plotHeight = canvas.height - margin.top - margin.bottom;
  const rawX = ((x - margin.left) / plotWidth) * 100;
  const rawY = 100 - ((y - margin.top) / plotHeight) * 100;
  state.selectedProbe = { x: Math.max(0, Math.min(100, rawX)), y: Math.max(0, Math.min(100, rawY)) };
  state.selectedExample = { x: [state.selectedProbe.x, state.selectedProbe.y], y: 0, source: 'probe' };
  updateSelectedInsight();
  renderSpace();
  renderNetwork();
  renderActivationMap();
}

function renderAll() {
  renderSpace();
  renderNetwork();
  renderActivationMap();
  renderLearning();
  updateSelectedInsight();
  updateMetrics();
}

function updateOneStep() {
  if (!state.network) return;
  const result = state.network.step(state.train, state.learningRate);
  state.updates += 1;
  state.trainingHistory.push({ update: state.updates, loss: result.loss, accuracy: result.accuracy });
  const testResult = state.network.evaluate(state.test);
  state.testHistory.push({ update: state.updates, loss: testResult.loss, accuracy: testResult.accuracy });
  updateMetrics();
  renderSpace();
  renderNetwork();
  renderActivationMap();
  updateSelectedInsight();
}

function trainLoop() {
  if (!state.running) return;
  if (state.updates >= 500) {
    state.running = false;
    updateMetrics();
    return;
  }
  updateOneStep();
  if (state.running) state.animationTimer = setTimeout(trainLoop, 35);
}

function startTraining() {
  if (state.running) return;
  state.running = true;
  if (state.updates >= 500) resetModel();
  $('#statusPill').classList.add('running');
  $('#statusPill').lastChild.textContent = ' Training';
  updateMetrics();
  trainLoop();
}

function pauseTraining() {
  state.running = false;
  if (state.animationTimer) clearTimeout(state.animationTimer);
  state.animationTimer = null;
  updateMetrics();
}

function setNeuronZero() {
  if (state.selectedNeuron?.type !== 'hidden') return;
  const selected = state.selectedExample;
  const old = state.network.forward(selected?.x ?? [50,50]).activated[state.selectedNeuron.index];
  state.neuronDisabled = state.selectedNeuron.index;
  const result = forwardForDisplay(selected?.x ?? [50,50]);
  const p = result.output[0];
  renderNetwork();
  renderActivationMap();
  renderSpace();
  $('#neuronExplanation').textContent = `Neuron ${state.selectedNeuron.index + 1} was temporarily zeroed. Its previous activation was ${old.toFixed(3)}.`;
  $('#outputValue').textContent = `Predicted p(B) with neuron zeroed: ${p.toFixed(3)}`;
}

function restoreNeuron() {
  state.neuronDisabled = null;
  renderNetwork();
  renderActivationMap();
  renderSpace();
  updateSelectedInsight();
}

function bindControls() {
  $('#presetSelect').addEventListener('change', event => {
    state.preset = event.target.value;
    loadDataset();
  });
  $('#hiddenSelect').addEventListener('change', event => {
    state.hiddenUnits = Number(event.target.value);
    loadDataset();
  });
  $('#learningRate').addEventListener('input', event => {
    state.learningRate = Number(event.target.value);
    $('#learningRateValue').textContent = state.learningRate.toFixed(3);
  });
  $('#learningRate').addEventListener('change', () => { resetModel(); });
  $('#seedInput').addEventListener('change', () => { loadDataset(); });
  $('#trainButton').addEventListener('click', startTraining);
  $('#pauseButton').addEventListener('click', pauseTraining);
  $('#stepButton').addEventListener('click', () => { if (!state.running) updateOneStep(); });
  $('#resetButton').addEventListener('click', () => resetModel());
  $('#repeatButton').addEventListener('click', () => { loadDataset(); });
  canvas.addEventListener('click', event => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * canvas.width / rect.width;
    const y = (event.clientY - rect.top) * canvas.height / rect.height;
    const nearest = state.train.concat(state.test).map(item => ({ item, distance: Math.hypot(item.x[0] - ((x - 45) / (canvas.width - 65) * 100), item.x[1] - (100 - (y - 20) / (canvas.height - 60) * 100)) })).sort((a,b)=>a.distance-b.distance)[0];
    if (nearest && nearest.distance < 8) withSelectedExample(nearest.item);
    else probeAtCanvas(x,y);
  });
  document.addEventListener('keydown', event => {
    if (event.key.toLowerCase() === 'p' && state.selectedNeuron?.type === 'hidden') setNeuronZero();
    if (event.key.toLowerCase() === 'r' && state.selectedNeuron?.type === 'hidden') restoreNeuron();
    if (event.key === ' ') { event.preventDefault(); state.running ? pauseTraining() : startTraining(); }
  });
  window.addEventListener('beforeunload', () => { if (state.animationTimer) clearTimeout(state.animationTimer); });
}

function initialize() {
  state.dataset = makeDataset(state.preset, state.seed);
  state.train = state.dataset.train;
  state.test = state.dataset.test;
  state.selectedExample = state.train[Math.floor(state.train.length / 2)];
  updateDatasetMeta();
  resetModel();
  bindControls();
  updateMetrics();
  renderAll();
}

initialize();
