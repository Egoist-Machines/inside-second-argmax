"""Two exact checks of the supplied sixteen-neuron checkpoint.

Run with Python's standard library only. Binary64 JSON numbers are converted
to their exact binary rational values, not interpreted as decimal fractions.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = '498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
SOURCE_MODEL = '14fb1fd8c97393ac996f9a19b224bfa433e05be5eb4e758d0c514ba41c047a9a'
EXACT_WEIGHTS = '5d58407f53ea03c0630a77d7a5d10e222d6b418c1e704ac191289ecebba8b3ce'
REFERENCE_CANDIDATE = '66d764e4fd9d27e36e5ff9ff90ae58cb7219dd9e53bb2411c1c5221d6c439b19'
REFERENCE_FALSE = '5aa7865f917c134d501e29a4037460dae161398347b883d3e4d2d37248fdcd8c'
P = (0, 3, 10, 11, 12, 13, 14, 15)
Q = tuple(i for i in range(16) if i not in (4, 5, 9))
C = (1, 6, 7, 8)
ZERO = F(0)
DOMAIN = (F(511, 512)-F(1, 16384), F(511, 512)+F(1, 16384))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def digest(value):
    raw = json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), ZERO)


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def at(offset, slope, value):
    return [a+b*value for a, b in zip(offset, slope)]


def load_model(path):
    data = json.loads(Path(path).read_text())
    require(data['checkpoint_sha256'] == CHECKPOINT and data['hidden_size'] == 16
            and data['sequence_length'] == 10, 'Checkpoint identity or dimensions changed')
    require(data.get('source_model_sha256', sha(path)) == SOURCE_MODEL, 'Source export identity changed')
    u = list(map(F, data['input']))
    W = [list(map(F, row)) for row in data['recurrent']]
    V = [list(map(F, row)) for row in data['readout']]
    require(len(u) == len(W) == 16 and len(V) == 10
            and all(len(row) == 16 for row in (*W, *V)), 'Expected all 432 native weights')
    require(digest(dict(input=u, recurrent=W, readout=V)) == EXACT_WEIGHTS, 'Exact native weights changed')
    return u, W, V


def candidate_source(u, W, V):
    """Finite ReLU secants retain all sixteen outgoing source paths."""
    G = dot(W[5], [max(ZERO, x) for x in u])
    K = sum((W[5][i]*u[i] for i in Q), ZERO)
    require(G > 0 and K > 0 and u[5] < 0, 'Candidate normalization signs changed')
    credit = [-sum((W[5][i]*W[i][j] for i in Q), ZERO)/K for j in range(16)]
    cutoff = [x/abs(u[5]) for x in W[5]]
    source = [G*row[5] for row in W]
    require(all(source[i] < 0 for i in P) and source[5] < 0, 'Reserve orientation changed')
    rows = {f'margin7_minus{j}': [a-b for a, b in zip(V[7], V[j])]
            for j in range(10) if j != 7}
    rows.update(c=credit, next_cutoff=cutoff)
    on = (0, 14, 15)
    free = [i for i in range(16) if i not in on]
    bounds = {}
    for name, row in rows.items():
        products = [a*b for a, b in zip(row, source)]
        fixed = sum((products[i] for i in on), ZERO)
        bounds[name] = dict(lower=fixed+sum((min(products[i], ZERO) for i in free), ZERO),
                            upper=fixed+sum((max(products[i], ZERO) for i in free), ZERO))
    require(all(bounds[f'margin7_minus{j}']['lower'] > 0 for j in range(10) if j != 7)
            and bounds['c']['lower'] > 0 and bounds['next_cutoff']['upper'] < 0,
            'The complete joint source contract lost a strict sign')
    # An independent enumeration checks the exact extrema over all 8192
    # free-secant corners against the signed interval formula.
    low, high = {name: None for name in rows}, {name: None for name in rows}
    vertices = [dict((name, sum((rows[name][i]*source[i] for i in on), ZERO)) for name in rows)]
    for i in free:
        vertices += [{name: value[name]+rows[name][i]*source[i] for name in rows} for value in vertices]
    for value in vertices:
        for name in rows:
            low[name] = value[name] if low[name] is None else min(low[name], value[name])
            high[name] = value[name] if high[name] is None else max(high[name], value[name])
    require(all((low[name], high[name]) == (bounds[name]['lower'], bounds[name]['upper']) for name in rows),
            'Independent corner enumeration disagrees')
    return dict(candidate_scale=G, eligibility_scale=K, credit_row=credit, cutoff_row=cutoff,
        physical_source_per_unit_candidate=source, forced_on_receivers=on, free_receivers=free,
        response_bounds=bounds, free_secant_corners=len(vertices), exact_gain_checks=len(vertices)*len(rows),
        premise='Two nonnegative old states differ only by G5*delta_a*e5, delta_a>0, at one shared input. PRE0/PRE14/PRE15 are nonnegative at the endpoint with the larger old candidate.',
        proof='The three source gains are negative, so that endpoint has the smaller PRE. Its nonnegative reserves force those secants to 1. Every other receiver has an independent secant in [0,1].',
        consequence='The complete source raises row 7 relative to every rival, raises c, and lowers the new cutoff. At any shared following input the lower cutoff weakly suppresses the following candidate.',
        scope='Conditional source increments, not an absolute winner or true-rank certificate. The other 13 gates may change. Secant-box corners need not be jointly reachable.')


def cell(pre0, pre1, domain):
    endpoints = [at(pre0, pre1, z) for z in domain]
    require(all(a*b > 0 for a, b in zip(*endpoints)), 'A physical gate changes inside the declared family')
    active = [i for i, value in enumerate(endpoints[0]) if value > 0]
    return ([value if i in active else ZERO for i, value in enumerate(pre0)],
            [value if i in active else ZERO for i, value in enumerate(pre1)],
            dict(active_rows=active, minimum_endpoint_gate_slack=min(abs(v) for row in endpoints for v in row),
                 preactivation_coefficients_sha256=digest([pre0, pre1])))


def patch_columns(u, W):
    common = [ZERO]*16
    contrast = [ZERO]*16
    for i in C:
        common[i] = abs(u[i])*W[7][4]/abs(u[7])
        contrast[i] = W[i][4]-common[i]
    require(contrast[7] == 0, 'A contrast cut changes reference comparator7')
    return common, contrast


def exact_trace(u, W, V, inputs, correction):
    old = [ZERO]*16
    gates = []
    for time, current in enumerate(inputs):
        pre = at(mv(W, old), u, current)
        if time == 3:
            pre = [value+change*old[4] for value, change in zip(pre, correction)]
        require(all(pre), 'A displayed fixed control has a degenerate gate')
        old = [max(ZERO, value) for value in pre]
        gates.append([i for i, value in enumerate(pre) if value > 0])
    scores = mv(V, old)
    winner = max(range(10), key=scores.__getitem__)
    return dict(winner=winner, scores=scores, active_rows=gates,
        minimum_winner_margin=min(scores[winner]-score for j, score in enumerate(scores) if j != winner))


def false_record(u, W, V):
    common, contrast = patch_columns(u, W)
    patches = dict(native=[ZERO]*16, common_query_cut=[-a for a in common], contrast_cut=[-a for a in contrast])
    old = [ZERO]*16
    prefix_gates = []
    for current in (F(1), ZERO):
        pre = at(mv(W, old), u, current)
        require(all(pre), 'The native two-input prefix has a degenerate gate')
        old = [max(ZERO, value) for value in pre]
        prefix_gates.append([i for i, value in enumerate(pre) if value > 0])
    threshold = -dot(W[4], old)/u[4]
    thresholds = {i: dot(W[i], old)/abs(u[i]) for i in C}
    require(threshold < DOMAIN[0] < DOMAIN[1] < 1 and max(thresholds[i] for i in C if i != 7) < DOMAIN[0]
            and DOMAIN[1] < thresholds[7], 'The record and partial-query ordering changed')
    arrival0, arrival1, arrival = cell(mv(W, old), u, DOMAIN)
    require(all(arrival0[i] == arrival1[i] == 0 for i in (1, 6, 8))
            and all(arrival0[i]+z*arrival1[i] > 0 for i in (4, 7) for z in DOMAIN), 'Unexpected false-offer gates')
    policies = {}
    all_masks = None
    expected = dict(native=0, common_query_cut=0, contrast_cut=2)
    for name, correction in patches.items():
        h0, h1 = arrival0[:], arrival1[:]
        suffix = []
        for time in range(3, 10):
            pre0, pre1 = mv(W, h0), mv(W, h1)
            if time == 3:
                pre0 = [a+b*arrival0[4] for a, b in zip(pre0, correction)]
                pre1 = [a+b*arrival1[4] for a, b in zip(pre1, correction)]
            h0, h1, record = cell(pre0, pre1, DOMAIN)
            suffix.append(dict(input_index=time, **record))
        scores0, scores1 = mv(V, h0), mv(V, h1)
        winner = expected[name]
        margins = {j: min(scores0[winner]-scores0[j]+z*(scores1[winner]-scores1[j]) for z in DOMAIN)
                   for j in range(10) if j != winner}
        require(all(value > 0 for value in margins.values()), 'An all-rival family margin is not strict')
        masks = [record['active_rows'] for record in suffix]
        require(all_masks is None or masks == all_masks, 'The three source policies do not share the proven suffix cells')
        all_masks = masks
        policies[name] = dict(winner=winner, all9_margin_lower=margins, suffix=suffix,
            margin2_minus0_intercept=scores0[2]-scores0[0], margin2_minus0_slope=scores1[2]-scores1[0],
            final_affine_scores_sha256=digest([scores0, scores1]))
    true_inputs = [F(1), ZERO, F(513, 512), *([ZERO]*7)]
    true_control = {name: exact_trace(u, W, V, true_inputs, correction) for name, correction in patches.items()}
    native_margin = true_control['native']['scores'][0]-true_control['native']['scores'][2]
    cut_margin = true_control['contrast_cut']['scores'][0]-true_control['contrast_cut']['scores'][2]
    require(true_control['native']['winner'] == true_control['contrast_cut']['winner'] == 0
            and native_margin > cut_margin > 0, 'The matched true-offer margin control changed')
    for value in true_control.values():
        require(value['minimum_winner_margin'] > 0, 'A true-control winner is not strict')
    return dict(false_domain=DOMAIN, center=F(511, 512), inputs='[1,0,z,0,0,0,0,0,0,0]',
        target=2, native_wrong_maximum=0, record_threshold=threshold, comparator_thresholds=thresholds,
        prefix_active_rows=prefix_gates, arrival=arrival, source_read_input_index=3,
        common_query_column_per_old_h4=common, contrast_column_per_old_h4=contrast, source_patches=patches, policies=policies,
        matched_true_offer=dict(inputs=true_inputs, target=0, native_margin0_minus2=native_margin,
            contrast_cut_margin0_minus2=cut_margin, policies=true_control),
        scope='Exact rational family and one matched control. The cut acts only on old record neuron 4 during update 3, then native ReLUs resume. It is not a uniformly helpful inference rule. The true control loses margin but retains its correct winner.')


def candidate_demo(u, W, V, candidate):
    prefix = [F(.8), ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, F(.2), ZERO]
    old = [ZERO]*16
    for current in prefix:
        old = [max(ZERO, value) for value in at(mv(W, old), u, current)]
    erased = old[:];erased[5] = ZERO
    full_pre, zero_pre = at(mv(W, old), u, F(1)), at(mv(W, erased), u, F(1))
    full = [max(ZERO, value) for value in full_pre]
    zero = [max(ZERO, value) for value in zero_pre]
    require(old[5] > 0 and all(full_pre[i] > 0 for i in (0, 14, 15)), 'The fixed demo lacks its supplied-source guard')
    scores, zero_scores = mv(V, full), mv(V, zero)
    margin = {j: scores[7]-scores[j]-zero_scores[7]+zero_scores[j] for j in range(10) if j != 7}
    delta = [a-b for a, b in zip(full, zero)]
    dc, dt = dot(candidate['credit_row'], delta), dot(candidate['cutoff_row'], delta)
    require(all(value > 0 for value in margin.values()) and dc > 0 and dt < 0, 'The fixed demo violates the joint source result')
    return dict(prefix=list(map(float, prefix)), prefix_exact=list(map(str, prefix)), current_input=1,
        source_index=5, retention_range=[0, 1], old_physical=list(map(float, old)), old_physical_exact_sha256=digest(old),
        candidate_scale=float(candidate['candidate_scale']), full_endpoint_pre=list(map(float, full_pre)),
        source_erased_scores=list(map(float, zero_scores)), source_full_scores=list(map(float, scores)),
        all9_margin_gains={str(j): float(value) for j, value in margin.items()}, c_gain=float(dc), tau_gain=float(dt),
        reserve_exact={str(i): str(full_pre[i]) for i in (0, 14, 15)},
        gain_exact=dict(all9_margin={str(j): str(value) for j, value in margin.items()}, c=str(dc), tau=str(dt)),
        scope='A fixed native nine-input prefix supplies the full old state. Retention scales only old physical neuron 5 while holding the other 15 coordinates fixed; erased states are internal controls, not alternate input histories. Displayed numbers round exact rational results to binary64.')


def reference_checks(candidate, false):
    compact = dict(domain=false['false_domain'], policies={name:dict(winner=value['winner'],
        all9_margin_lower=value['all9_margin_lower'], active_rows=[row['active_rows'] for row in value['suffix']])
        for name,value in false['policies'].items()}, true_policies={name:dict(winner=value['winner'],scores=value['scores'])
        for name,value in false['matched_true_offer']['policies'].items()})
    require(digest(candidate['response_bounds']) == REFERENCE_CANDIDATE, 'Preserved candidate bounds differ')
    require(digest(compact) == REFERENCE_FALSE, 'Preserved record-family certificate differs')
    return dict(candidate_bounds_sha256=REFERENCE_CANDIDATE, record_family_selection_sha256=REFERENCE_FALSE,
        compared_exactly=dict(candidate_bounds=22, false_family_margin_bounds=27, suffix_masks=21, true_control_scores=30),
        source_certificates=dict(candidate_history='20548970632b17f693dfb014f90685c9a59ed392aa72e8a4d54a498f66f03350',
            semantic_record_phase_program='e461c5fbaf18cff6a4f473b021ae3f9b6b9f1796a2800e3e0d1a6acece5c399c'))


def verify(model):
    u, W, V = load_model(model)
    candidate, false = candidate_source(u, W, V), false_record(u, W, V)
    return dict(schema_version=1, valid=True, checkpoint_sha256=CHECKPOINT, source_model_sha256=SOURCE_MODEL,
        model_json_sha256=sha(model), exact_weights_sha256=EXACT_WEIGHTS, verifier_sha256=sha(__file__),
        arithmetic='Exact rational arithmetic on binary64 native weights. This is not a certificate of floating-point rounding.',
        candidate_source=candidate, false_record=false, candidate_demo=candidate_demo(u,W,V,candidate),
        preserved_reference_checks=reference_checks(candidate,false))


def examples(model, result):
    u,W,V = load_model(model);false=result['false_record'];center=[F(1),ZERO,F(511,512),*([ZERO]*7)]
    points={name:exact_trace(u,W,V,center,patch) for name,patch in false['source_patches'].items()}
    return dict(schema_version=1,model_json_sha256=sha(model),checkpoint_sha256=CHECKPOINT,
        candidate_demo=result['candidate_demo'],false_record_demo=dict(false_inputs=list(map(float,center)),
            true_inputs=[1,0,float(F(513,512)),*([0]*7)],offer_interval=list(map(float,DOMAIN)),
            offer_interval_exact=list(map(str,DOMAIN)),patch_input_index=3,source_index=4,comparators=list(C),
            common_query_correction_per_old_h4=list(map(float,false['source_patches']['common_query_cut'])),
            contrast_correction_per_old_h4=list(map(float,false['source_patches']['contrast_cut'])),
            correction_exact=encode(false['source_patches']),
            default_predictions={name:value['winner'] for name,value in points.items()},
            default_scores={name:list(map(float,value['scores'])) for name,value in points.items()},
            true_margin_control=dict(native=float(false['matched_true_offer']['native_margin0_minus2']),
                contrast_cut=float(false['matched_true_offer']['contrast_cut_margin0_minus2']),winner=0,target=0),
            patch_rule='At input index 3 only, add correction_per_old_h4[i]*old_h4 to each physical PRE_i before ReLU. Keep all other sources, then resume native updates.',
            scope=false['scope']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'results/model.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', type=Path, help='Compare a saved proof with exact recomputation')
    parser.add_argument('--examples', type=Path, help='Write the two fixed demonstration fixtures')
    args = parser.parse_args(argv)
    exact_result = verify(args.model);result = encode(exact_result)
    if args.check:
        require(json.loads(args.check.read_text()) == result, 'Saved proof differs from exact recomputation')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    elif args.check:
        print('Verified both exact proofs and the supplied-candidate demonstration.')
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    if args.examples:
        args.examples.write_text(json.dumps(encode(examples(args.model,exact_result)),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
