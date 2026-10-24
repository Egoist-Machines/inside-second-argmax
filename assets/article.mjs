import {recordExample, traceExample, writeExample} from './model.mjs?v=3';

const $ = selector => document.querySelector(selector);
const signed = value => Math.abs(value) < 1e-12 ? '0' : `${value > 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;

function row(label, value, low, high, selection = null) {
  const element = document.createElement('div');
  element.className = 'bar-row';
  const name = document.createElement('span');
  name.className = 'policy-label';
  name.textContent = label;
  if (selection !== null) {
    const detail = document.createElement('small');
    detail.textContent = `Selects ${selection}`;
    name.append(detail);
  }
  const track = document.createElement('span');
  track.className = 'bar-track';
  track.setAttribute('aria-hidden', 'true');
  const zero = -low / (high - low) * 100;
  if (low < 0) {
    const line = document.createElement('i');
    line.className = 'zero-line';
    line.style.left = `${zero}%`;
    track.append(line);
  }
  const bar = document.createElement('i');
  bar.className = `bar-fill${value < 0 ? ' negative' : ''}`;
  bar.style.left = `${Math.min(zero, (value - low) / (high - low) * 100)}%`;
  bar.style.width = `${Math.abs(value) / (high - low) * 100}%`;
  track.append(bar);
  const number = document.createElement('span');
  number.className = 'value';
  number.textContent = signed(value);
  element.append(name, track, number);
  return element;
}

async function initialise() {
  const response = await fetch(new URL('../results/model.json', import.meta.url));
  if (!response.ok) throw new Error('The checkpoint data could not be loaded.');
  const model = await response.json();
  const traceEndpoints = [traceExample(model, .25), traceExample(model, 2)];
  const traceValues = traceEndpoints.flatMap(item => [item.comparisonMargin, item.otherMargin, item.margin]);
  const traceLow = Math.min(0, ...traceValues) * 1.1;
  const traceHigh = Math.max(0, ...traceValues) * 1.1;

  function updateTrace() {
    const isRecord = $('input[name="trace-case"]:checked').value === 'record';
    const result = traceEndpoints[Number(isRecord)];
    $('#trace-chart').replaceChildren(
      row('Comparisons', result.comparisonMargin, traceLow, traceHigh),
      row('Other neurons', result.otherMargin, traceLow, traceHigh),
      row('Total', result.margin, traceLow, traceHigh));
    $('#trace-selection').textContent = `Selects position ${result.prediction}, value ${result.inputs[result.prediction]}.`;
  }

  const fullWrite = writeExample(model, 1);
  if (!fullWrite.reserve.every(item => item.pre >= 0)) throw new Error('The saved write example does not satisfy its carrier premise.');
  const writeMax = Math.max(...fullWrite.gains.map(item => item.gain)) * 1.06;

  function updateWrite() {
    const percentage = Number($('#retention').value);
    const result = writeExample(model, percentage / 100);
    $('#retention-value').textContent = `${percentage}%`;
    $('#retention').setAttribute('aria-valuetext', `${percentage}%`);
    $('#write-chart').replaceChildren(...result.gains.map(item => row(`vs ${item.rival}`, item.gain, 0, writeMax)));
    $('#base-change').textContent = signed(result.c);
    $('#cutoff-change').textContent = signed(result.tau);
  }

  function updateRecord() {
    const trueRecord = $('input[name="offer-case"]:checked').value === 'true';
    $('#offer').disabled = trueRecord;
    $('#offer').hidden = trueRecord;
    const z = trueRecord ? 513 / 512 : 511 / 512 + Number($('#offer').value) / 16384;
    const result = recordExample(model, z);
    $('#offer-value').textContent = z.toFixed(8);
    $('#offer').setAttribute('aria-valuetext', z.toFixed(8));
    $('#record-margin-label').textContent = trueRecord ? 'Score 0 − score 2' : 'Score 2 − score 0';
    const endpoints = trueRecord ? [result] : [-1, 1].map(e => recordExample(model, 511 / 512 + e / 16384));
    const values = endpoints.flat().map(item => item.margin);
    const low = Math.min(0, ...values) * 1.12;
    const high = Math.max(0, ...values) * 1.12;
    $('#record-chart').replaceChildren(...result.map(item => row(item.label, item.margin, low, high, item.prediction)));
  }

  $('#retention').addEventListener('input', updateWrite);
  $('#offer').addEventListener('input', updateRecord);
  document.querySelectorAll('input[name="offer-case"]').forEach(input => input.addEventListener('change', updateRecord));
  document.querySelectorAll('input[name="trace-case"]').forEach(input => input.addEventListener('change', updateTrace));
  updateTrace();
  updateWrite();
  updateRecord();
}

initialise().catch(error => {
  for (const id of ['#trace-chart', '#write-chart', '#record-chart']) {
    const message = document.createElement('p');
    message.className = 'load-error';
    message.textContent = error.message;
    $(id).replaceChildren(message);
  }
});

$('#copy-citation').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('#citation-text').textContent);
    $('#citation-status').textContent = 'Copied';
  } catch {
    $('#citation-status').textContent = 'Select the text above to copy it.';
  }
});
