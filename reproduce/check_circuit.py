"""Check all parameter columns and full prefix trajectories using NumPy only.

Run: python reproduce/check_circuit.py
Optional: --archives path/to/controlled_inputs67.json path/to/controlled_inputs68.json
The optional archives are already-used regression inputs, not fresh validation.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

if __package__:
    from .circuit import Circuit, P, Q, signed_caps
else:
    from circuit import Circuit, P, Q, signed_caps

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear(rows, values):
    """Independent physical reference: fixed left-to-right column accumulation."""
    out = np.zeros((*values.shape[:-1], len(rows)), dtype=np.float64)
    for j in range(rows.shape[1]):
        out += values[..., j, None]*rows[:, j]
    return out


def coordinates(model):
    """Derive independent encoder rows using exact binary64 fractions."""
    u = [F(v) for v in model['input']]
    W = [[F(v) for v in row] for row in model['recurrent']]
    dot = lambda a, b: sum((x*y for x, y in zip(a, b)), F(0))
    G = dot(W[5], [max(F(0), v) for v in u]); G9 = dot(W[9], [max(F(0), v) for v in u])
    memory = -u[4]/W[4][2]; K = sum((W[5][i]*u[i] for i in Q), F(0))
    costs = [-W[i][5]*G for i in P]
    c = [-sum((W[5][i]*W[i][j] for i in Q), F(0))/K for j in range(16)]
    H = [(-W[4][j]/u[4]+W[7][j]/abs(u[7]))/2 for j in range(16)]
    L = [W[4][j]/u[4]+W[7][j]/abs(u[7]) for j in range(16)]
    tau = [v/abs(u[5]) for v in W[5]]
    native = [[F(0)]*16 for _ in range(8)]
    native[0][2], native[1][4], native[2][7] = 1/memory, 1/u[4], 1/abs(u[7])
    for at, i in enumerate((1, 6, 8), 3):
        native[at][7], native[at][i] = 1/abs(u[7]), -1/abs(u[i])
    native[6][5], native[7][9] = 1/G, 1/G9
    widths = [[a-b for a, b in zip(native[4], native[3])], [-v for v in native[5]]]
    z = []
    for i, cost in zip(P, costs):
        row = [F(0)]*16; row[i] = 1/cost; z.append(row)
    array = lambda rows: np.array([[float(v) for v in row] for row in rows])
    return array([c, H, L, tau, *widths, native[6], native[7], *z]), array(native), array(z)


def coefficient_check(circuit, model, encode, native, capacity):
    u, W, V = (np.array(model[k]) for k in ('input', 'recurrent', 'readout'))
    eye = np.eye(16)
    _, columns = circuit.step(eye@encode.T, np.zeros(16))
    _, first = circuit.step(circuit.zero(1), np.ones(1))
    f, z = eye@native.T, eye@capacity.T
    generated, _ = signed_caps(f[:, 2], f[:, 3:6])
    recovered_V = circuit._scores(f, z, generated).T
    contexts = np.vstack((np.zeros((1, 16)), eye, np.arange(1, 17)[None]/16))
    physical = np.repeat(contexts, 5, axis=0); x = np.tile([-2., -.5, 0., .5, 2.], len(contexts))
    state, row = circuit.step(physical@encode.T, x)
    expected_pre = linear(W, physical)+x[:, None]*u; expected_h = np.maximum(expected_pre, 0.)
    errors = dict(input=float(abs(first['pre'][0]-u).max()), recurrent=float(abs(columns['pre'].T-W).max()),
        readout=float(abs(recovered_V-V).max()), physical_roundtrip=float(abs(circuit.restore(physical@encode.T)-physical).max()),
        context_pre=float(abs(row['pre']-expected_pre).max()), context_physical=float(abs(row['physical']-expected_h).max()),
        context_state=float(abs(state-expected_h@encode.T).max()), context_scores=float(abs(row['scores']-linear(V, expected_h)).max()))
    return dict(parameters=432, input_coefficients=16, recurrent_coefficients=256, readout_coefficients=160,
        independent_one_step_contexts=90, max_abs_errors=errors, within_tolerance=all(v <= TOLERANCE for v in errors.values()))


def physical_trace(x, u, W, V):
    h = np.zeros((len(x), 16)); hs, ps, scores = [], [], []
    for column in x.T:
        pre = linear(W, h)+column[:, None]*u
        h = np.maximum(pre, 0.)
        hs.append(h.copy()); ps.append(pre); scores.append(linear(V, h))
    return np.stack(hs, 1), np.stack(ps, 1), np.stack(scores, 1)


def compare(circuit, x, model, encode, batch=256):
    u, W, V = (np.array(model[k]) for k in ('input', 'recurrent', 'readout'))
    errors = dict(state=0., physical=0., pre=0., scores=0., recovery_adjustment=0., available=0., spent=0.)
    counts = dict(prefix_choice_differences=0, proposal_sign_differences=0, proposal_zero_differences=0,
                  decoded_activity_differences=0, decoded_negative_entries=0)
    activity_by_unit = np.zeros(16, dtype=np.int64)
    max_inactive_residue = 0.; max_lost_active = 0.
    for start in range(0, len(x), batch):
        xx = x[start:start+batch]
        states, scores, row = circuit.forward(xx)
        h, pre, gold = physical_trace(xx, u, W, V)
        for key, value in dict(state=states-h@encode.T, physical=row['physical']-h,
            pre=row['pre']-pre, scores=scores-gold, recovery_adjustment=row['recovery_adjustment']).items():
            errors[key] = max(errors[key], float(abs(value).max()))
        old = np.concatenate((np.zeros_like(h[:, :1]), h[:, :-1]), axis=1)
        supply_rows = W[list(P)].copy(); supply_rows[:, 5] = 0
        background = (linear(supply_rows, old)+xx[..., None]*u[list(P)])/circuit.k.candidate_cost
        available = np.maximum(background, 0.)
        spent = np.minimum(available, old[..., 5:6]/circuit.k.candidate_scale)
        errors['available'] = max(errors['available'], float(abs(row['available']-available).max()))
        errors['spent'] = max(errors['spent'], float(abs(row['spent']-spent).max()))
        counts['prefix_choice_differences'] += int((scores.argmax(-1) != gold.argmax(-1)).sum())
        counts['proposal_sign_differences'] += int(((row['pre'] > 0) != (pre > 0)).sum())
        counts['proposal_zero_differences'] += int(((row['pre'] == 0) != (pre == 0)).sum())
        activity = (row['physical'] > 0) != (h > 0)
        counts['decoded_activity_differences'] += int(activity.sum())
        activity_by_unit += activity.sum((0, 1))
        max_inactive_residue = max(max_inactive_residue, float(np.max(np.where(h == 0, abs(row['physical']), 0), initial=0)))
        max_lost_active = max(max_lost_active, float(np.max(np.where((h > 0) & (row['physical'] <= 0), h, 0), initial=0)))
        counts['decoded_negative_entries'] += int((row['physical'] < 0).sum())
    return dict(sequences=len(x), steps=x.shape[1], max_abs_errors=errors, **counts,
        decoded_activity_differences_by_unit=activity_by_unit.tolist(),
        max_decoded_magnitude_when_native_zero=max_inactive_residue, max_native_activity_lost_in_decode=max_lost_active,
        numeric_and_choice_agreement=all(v <= TOLERANCE for v in errors.values())
            and counts['prefix_choice_differences'] == 0
            and counts['proposal_sign_differences'] == 0
            and counts['proposal_zero_differences'] == 0)


def examples():
    named = [
        ('mixed', [.3, -.4, 1.2, .5, -.2, .8, .1, -.6, 1.7, .4]),
        ('false_record', [1., 0., 511/512, 0., 0., 0., 0., 0., 0., 0.]),
        ('negative_flat', [-1.]*10),
        ('negative_swap', np.array([-64, -26, -20, -13, -36, -38, -33, -39, -44, -63])/64),
        ('supplied_candidate', [.8, 0., 0., 0., 0., 0., 0., .2, 0., 1.]),
        ('zero', [0.]*10),
    ]
    for sign in (1., -1.):
        for i in range(10):
            x = np.zeros(10); x[i] = sign; named.append((f'impulse_{sign:+g}_{i}', x))
    return [name for name, _ in named], np.array([x for _, x in named], dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path, help='Write a new report; default leaves saved results unchanged.')
    args = parser.parse_args(argv)
    table_path, model_path = ROOT/'results/circuit.json', ROOT/'results/model.json'
    model = json.loads(model_path.read_text()); circuit = Circuit.from_json(table_path)
    if circuit.record['checkpoint_sha256'] != model['checkpoint_sha256'] or circuit.record['source_model_sha256'] != model['source_model_sha256']:
        raise ValueError('Model/table source identity mismatch')
    encode, native, capacity = coordinates(model)
    coverage = coefficient_check(circuit, model, encode, native, capacity)
    names, x = examples(); fixture = compare(circuit, x, model, encode)
    fixture['names'] = names
    _, scores, _ = circuit.forward(x)
    fixture['inputs'], fixture['final_choices'] = x.tolist(), scores[:, -1].argmax(-1).tolist()
    archives = []
    for path in args.archives:
        saved = json.loads(path.read_text()); values = np.array(saved['inputs'], dtype=np.float64)
        if values.shape != (saved['parents'], 10) or hashlib.sha256(values.astype('<f8').tobytes()).hexdigest() != saved['little_f8_sha256']:
            raise ValueError('Archive input identity mismatch')
        result = compare(circuit, values, model, encode)
        result.update(seed=saved['seed'], archive_sha256=sha(path), input_value_sha256=saved['little_f8_sha256'])
        archives.append(result)
    passed = coverage['within_tolerance'] and fixture['numeric_and_choice_agreement'] and all(a['numeric_and_choice_agreement'] for a in archives)
    report = dict(schema_version=1, status='PASS' if passed else 'FAIL',
        scope='Numerical reproduction on selected examples and previously evaluated inputs, not fresh confirmation.',
        checkpoint_sha256=model['checkpoint_sha256'], model_sha256=sha(model_path), table_sha256=sha(table_path),
        source_model_sha256=model['source_model_sha256'],
        table_payload_sha256=circuit.record['table_payload_sha256'], reference_algorithm_sha256=circuit.record['reference_algorithm_sha256'],
        fixed_tables=dict(arrays=len(circuit.record['tables']),stored_numbers=sum(np.size(v) for v in circuit.record['tables'].values()),
            note='Fixed derived coefficients, not additional learned parameters.'),
        source_sha256={p.name:sha(p) for p in (Path(__file__), ROOT/'reproduce/circuit.py', ROOT/'reproduce/requirements.txt')},
        environment=dict(python=platform.python_version(), numpy=np.__version__), absolute_tolerance=TOLERANCE,
        arithmetic='Float64 numerical agreement, not bitwise equivalence. All three signed head clips and stored-realization correction remain; merged-cap dispersion is implicit.',
        decoded_activity='The boundary decoder can leave tiny positive values where a native rectifier is zero. Counts are literal, with magnitudes retained. Generated events use proposal signs; decoded activity differences are not silently relabelled.',
        coefficient_check=coverage, fixtures=fixture, archived_regressions=archives)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fixture_summary = {k:v for k,v in fixture.items() if k not in ('inputs', 'names', 'final_choices')}
    print(json.dumps(dict(status=report['status'],coefficient_check=coverage,fixtures=fixture_summary,
        archived_regressions=archives),indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
