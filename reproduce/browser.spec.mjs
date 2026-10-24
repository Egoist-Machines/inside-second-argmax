import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {argmax, forward, recordExample, recordSources, traceExample, writeExample} from '../assets/model.mjs';

const model = JSON.parse(await readFile(new URL('../results/model.json', import.meta.url), 'utf8'));
const examples = JSON.parse(await readFile(new URL('../results/examples.json', import.meta.url), 'utf8'));
const trace = JSON.parse(await readFile(new URL('../results/successful-trace.json', import.meta.url), 'utf8'));
const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) < tolerance,
  `${actual} differs from ${expected}`);

test('native first-maximum convention', () => {
  assert.equal(argmax([2, 2, 0]), 0);
  assert.equal(argmax([-1, 0, 0]), 1);
  assert.equal(forward(model, Array(10).fill(0)).prediction, 0);
});

test('native record switch matches the independent scalar trace and full score decomposition', () => {
  for (const gold of trace.cases.filter(item => item.native)) {
    const actual = traceExample(model, gold.inputs.at(-1));
    assert.equal(actual.prediction, gold.target);
    close(actual.margin, gold.margin_0_minus_7);
    close(actual.comparisonMargin, gold.comparison_margin_0_minus_7);
    close(actual.otherMargin, gold.other_margin_0_minus_7);
    close(actual.comparisonMargin + actual.otherMargin, actual.margin);
    actual.scores.forEach((value, i) => close(value, gold.scores[i]));
    actual.hidden.forEach((value, i) => close(value, gold.hidden[i]));
  }
});

test('restoring the lower-input comparisons prevents the record switch', () => {
  const lower = traceExample(model, .25), record = traceExample(model, 2);
  const restored = record.hidden.slice();
  for (const i of trace.comparison_neurons) {
    assert.ok(lower.hidden[i] > 0);
    assert.equal(record.hidden[i], 0);
    restored[i] = lower.hidden[i];
  }
  const scores = model.readout.map(row => row.reduce((sum, value, i) => sum + value * restored[i], 0));
  assert.equal(argmax(scores), 7);
  scores.forEach((value, i) => close(value, trace.cases[2].scores[i]));
});

test('candidate example matches the independent rational witness', () => {
  const actual = writeExample(model, 1);
  const gold = examples.candidate_demo;
  assert.ok(actual.reserve.every(item => item.pre > 0));
  assert.ok(actual.gains.every(item => item.gain > 0));
  assert.ok(actual.c > 0 && actual.tau < 0);
  close(actual.c, gold.c_gain);
  close(actual.tau, gold.tau_gain);
  for (const item of actual.gains) close(item.gain, gold.all9_margin_gains[String(item.rival)]);
});

test('retention control erases exactly the selected source at zero', () => {
  const actual = writeExample(model, 0);
  assert.ok(actual.gains.every(item => Math.abs(item.gain) < 1e-12));
  close(actual.c, 0);
  close(actual.tau, 0);
});

test('finite retained writes satisfy the joint directional law', () => {
  for (const fraction of [.1, .25, .5, .75, 1]) {
    const actual = writeExample(model, fraction);
    assert.ok(actual.reserve.every(item => item.pre > 0));
    assert.ok(actual.gains.every(item => item.gain > 0));
    assert.ok(actual.c > 0 && actual.tau < 0);
  }
});

test('source split matches exact fixture vectors and leaves row7 contrast zero', () => {
  const actual = recordSources(model), gold = examples.false_record_demo;
  assert.ok(actual.contrast[7] === 0);
  for (let i = 0; i < 16; i++) {
    close(actual.common[i], gold.common_query_correction_per_old_h4[i]);
    close(actual.contrast[i], gold.contrast_correction_per_old_h4[i]);
  }
});

test('false-record endpoints and interior preserve the certified outcomes', () => {
  for (const epsilon of [-1, -.5, 0, .5, 1]) {
    const rows = recordExample(model, 511 / 512 + epsilon / 16384);
    assert.deepEqual(rows.map(row => row.prediction), [0, 0, 2]);
    assert.ok(rows[0].margin < 0 && rows[1].margin < 0 && rows[2].margin > 0);
    for (const row of rows) {
      assert.ok(row.scores.every((score, i) => i === row.prediction || row.scores[row.prediction] > score));
    }
  }
});

test('true-record contrast cut reduces margin without changing the winner', () => {
  const rows = recordExample(model, 513 / 512);
  assert.ok(rows.every(row => row.prediction === 0));
  assert.ok(rows[0].margin > rows[2].margin && rows[2].margin > 0);
});
