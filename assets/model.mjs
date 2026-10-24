export const CARRIERS = [0, 3, 10, 11, 12, 13, 14, 15];
export const COMPARATORS = [1, 6, 7, 8];

export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const argmax = values => values.reduce((best, value, i) => value > values[best] ? i : best, 0);

export function step(model, old, x, correction = null, source = 4) {
  const pre = model.recurrent.map((row, i) => dot(row, old) + model.input[i] * x
    + (correction ? correction[i] * old[source] : 0));
  const hidden = pre.map(value => Math.max(0, value));
  return {pre, hidden, scores: model.readout.map(row => dot(row, hidden))};
}

export function forward(model, inputs, correction = null) {
  let hidden = Array(16).fill(0);
  const states = [];
  const proposals = [];
  let result;
  for (const [t, x] of inputs.entries()) {
    result = step(model, hidden, x, t === 3 ? correction : null);
    hidden = result.hidden;
    states.push(hidden);
    proposals.push(result.pre);
  }
  return {...result, states, proposals, prediction: argmax(result.scores)};
}

export function recordSources(model) {
  const common = Array(16).fill(0), contrast = Array(16).fill(0);
  for (const i of COMPARATORS) {
    const q = Math.abs(model.input[i]) * model.recurrent[7][4] / Math.abs(model.input[7]);
    common[i] = -q;
    contrast[i] = -(model.recurrent[i][4] - q);
  }
  return {common, contrast};
}

export function recordExample(model, z) {
  const inputs = [1, 0, z, 0, 0, 0, 0, 0, 0, 0];
  const corrections = recordSources(model);
  const target = z > 1 ? 0 : 2;
  const rival = z > 1 ? 2 : 0;
  return [
    ['Original', forward(model, inputs)],
    ['Remove common query', forward(model, inputs, corrections.common)],
    ['Remove contrast', forward(model, inputs, corrections.contrast)]
  ].map(([label, result]) => ({label, prediction: result.prediction,
    margin: result.scores[target] - result.scores[rival], scores: result.scores, target, rival}));
}

export function traceExample(model, lastInput) {
  const inputs = [1, 0, 0, 0, 0, 0, 0, 0.5, 0, lastInput];
  const result = forward(model, inputs);
  const parts = Array.from({length: 16}, (_, i) =>
    (model.readout[0][i] - model.readout[7][i]) * result.hidden[i]);
  return {...result, inputs,
    comparisonMargin: parts.reduce((sum, value, i) => sum + (COMPARATORS.includes(i) ? value : 0), 0),
    otherMargin: parts.reduce((sum, value, i) => sum + (COMPARATORS.includes(i) ? 0 : value), 0),
    margin: result.scores[0] - result.scores[7]
  };
}

export function writeExample(model, retention) {
  const prefix = [0.8, 0, 0, 0, 0, 0, 0, 0.2, 0];
  const old = forward(model, prefix).hidden;
  const erased = old.slice();
  erased[5] = 0;
  const retained = old.slice();
  retained[5] *= retention;
  const baseline = step(model, erased, 1);
  const result = step(model, retained, 1);
  const Q = Array.from({length: 16}, (_, i) => i).filter(i => ![4, 5, 9].includes(i));
  const K = Q.reduce((sum, i) => sum + model.recurrent[5][i] * model.input[i], 0);
  const cRow = Array.from({length: 16}, (_, j) =>
    -Q.reduce((sum, i) => sum + model.recurrent[5][i] * model.recurrent[i][j], 0) / K);
  const difference = result.hidden.map((value, i) => value - baseline.hidden[i]);
  return {
    gains: Array.from({length: 10}, (_, rival) => rival).filter(i => i !== 7).map(rival => ({rival,
      gain: result.scores[7] - result.scores[rival] - baseline.scores[7] + baseline.scores[rival]})),
    c: dot(cRow, difference),
    tau: dot(model.recurrent[5], difference) / Math.abs(model.input[5]),
    reserve: [0, 14, 15].map(i => ({neuron: i, pre: result.pre[i]}))
  };
}
