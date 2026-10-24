"""Sixteen-state circuit: comparisons, admission, carrier supply and full scores.

State order: c,H,L,tau,w16,w87,a,b,z0,z3,z10,z11,z12,z13,z14,z15.
All proposals read the same old state. NumPy float64 is used throughout.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

P = (0, 3, 10, 11, 12, 13, 14, 15)
Q = tuple(i for i in range(16) if i not in (4, 5, 9))
J = tuple(Q.index(i) for i in Q if i not in (1, 6, 7, 8))
PAIR, TRIAD, PULSE, OUTER = (2, 3), (4, 5, 6), (1, 7), (0, 1, 4, 5, 6, 7)


def relu(x):
    return np.maximum(x, 0.)


def pair(common, difference):
    lo, hi = common-difference/2, common+difference/2
    return relu(lo), relu(hi), lo, hi


def signed_caps(debt, caps):
    plus, minus = relu(caps), relu(-caps)
    spent = np.minimum(relu(debt)[:, None], plus)
    released = np.minimum(relu(-debt)[:, None], minus)
    return spent-(minus-released), -(plus-spent)+released


class Circuit:
    def __init__(self, record):
        self.record = record
        values = {name: np.array(value, dtype=np.float64) for name, value in record['tables'].items()}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite circuit table')
        for value in values.values():
            value.flags.writeable = False
        self.k = SimpleNamespace(**values)

    @classmethod
    def from_json(cls, path=None):
        path = Path(path) if path is not None else Path(__file__).resolve().parents[1]/'results/circuit.json'
        record = json.loads(path.read_text())
        payload = json.dumps(record['tables'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if record['schema_version'] != 1 or hashlib.sha256(payload).hexdigest() != record['table_payload_sha256']:
            raise ValueError('Circuit table identity mismatch')
        return cls(record)

    def zero(self, batch):
        if type(batch) is not int or batch <= 0:
            raise ValueError('Expected positive batch size')
        return np.zeros((batch, 16), dtype=np.float64)

    def _read(self, state):
        s = np.asarray(state, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 16 or not len(s) or not np.isfinite(s).all() or (s[:, 6:] < 0).any():
            raise ValueError('Expected finite state with nonnegative candidate, relay and capacities')
        k = self.k; bank, z = s[:, :8], s[:, 8:]
        wide = np.zeros_like(s)
        wide[:, 3:5], wide[:, 6:8], wide[:, 8:] = bank[:, 4:6], bank[:, 6:8], z
        mdqv = (bank[:, :4]-wide@k.field_rows.T)@k.field_inverse.T
        qmin = np.maximum.reduce((np.zeros(len(s)), mdqv[:, 3]+abs(bank[:, 4])/2, -bank[:, 5]))
        scale = np.maximum(np.concatenate((abs(bank), abs(z), abs(mdqv)), axis=1).max(1), 1)
        tolerance = 128*np.finfo(np.float64).eps*scale
        if not np.isfinite(mdqv).all() or (mdqv[:, :2] < -tolerance[:, None]).any() or (mdqv[:, 2] < qmin-tolerance).any():
            raise ValueError('State outside the physical comparison cone')
        before = mdqv.copy()
        mdqv[:, :2] = relu(mdqv[:, :2]); mdqv[:, 2] = np.maximum(mdqv[:, 2], qmin)
        wide[:, :3], wide[:, 5] = mdqv[:, :3], mdqv[:, 3]
        controls = wide[:, :8].copy(); controls[:, 3:6] = wide[:, 3:6]@k.width_inverse.T
        return controls, z, wide, mdqv-before

    def _physical(self, f, z):
        k = self.k; h = np.zeros((len(f), 16))
        h[:, 2], h[:, 4], h[:, 7] = k.memory_scale*f[:, 0], k.record_input_scale*f[:, 1], k.comparator_scales[2]*f[:, 2]
        h[:, [1, 6, 8]] = (f[:, 2:3]-f[:, 3:6])*k.comparator_scales[[0, 1, 3]]
        h[:, 5], h[:, 9], h[:, P] = k.candidate_scale*f[:, 6], k.relay_scale*f[:, 7], z*k.candidate_cost
        return h

    def restore(self, state):
        f, z, _, _ = self._read(state)
        return self._physical(f, z)

    def split(self, physical):
        h = np.asarray(physical, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != 16 or not np.isfinite(h).all() or (h < 0).any():
            raise ValueError('Expected nonnegative physical states')
        k = self.k; q = h[:, 7]/k.comparator_scales[2]
        f = np.column_stack((h[:, 2]/k.memory_scale, h[:, 4]/k.record_input_scale, q,
            q-h[:, 1]/k.comparator_scales[0], q-h[:, 6]/k.comparator_scales[1],
            q-h[:, 8]/k.comparator_scales[3], h[:, 5]/k.candidate_scale, h[:, 9]/k.relay_scale))
        z = h[:, P]/k.candidate_cost; views = f[:, 3:6]@k.width_change.T
        wide = np.column_stack((f, z)); wide[:, 3:6] = views
        return np.column_stack((wide@k.field_rows.T, views[:, :2], f[:, 6:8], z))

    def _carriers(self, f, z, x):
        k = self.k; refill_scale = k.record_refill[list(PAIR)]
        paired = z[:, PAIR]/refill_scale
        ports = np.column_stack((paired.mean(1), paired[:, 1]-paired[:, 0]))
        transport = np.empty_like(z); transport[:, 0] = (z*k.pool_weights).sum(1)
        transport[:, PULSE] = (z[:, PULSE]@k.pulse_internal.T+z[:, 0:1]*k.pulse_pool
            + z[:, TRIAD]@k.pulse_triad.T+ports@k.pulse_ports.T)
        transport[:, TRIAD] = z[:, TRIAD]@k.triad_internal.T+z[:, [0, 1, 7]]@k.triad_outer.T+ports@k.triad_ports.T
        p = ports@k.pair_internal.T+z[:, OUTER]@k.pair_outer.T
        transport[:, 2], transport[:, 3] = refill_scale[0]*(p[:, 0]-p[:, 1]/2), refill_scale[1]*(p[:, 0]+p[:, 1]/2)
        memory, query = f[:, 0:1]*k.source_profiles[:, 0], f[:, 2:3]*k.source_profiles[:, 2]
        phase, relay = f[:, 3:6]@k.source_profiles[:, 3:6].T, f[:, 7:8]*k.source_profiles[:, 7]
        input_, refill = x[:, None]*k.input_profile, f[:, 1:2]*k.record_refill
        calibration = memory+query+phase+relay+input_
        pre = transport+calibration+refill-f[:, 6:7]
        pc = calibration[:, PAIR]/refill_scale
        common = p[:, 0]+pc.mean(1)+f[:, 1]-(1/refill_scale).mean()*f[:, 6]
        difference = p[:, 1]+pc[:, 1]-pc[:, 0]-(1/refill_scale[1]-1/refill_scale[0])*f[:, 6]
        lo, hi, pl, ph = pair(common, difference)
        pre[:, 2], pre[:, 3] = refill_scale[0]*pl, refill_scale[1]*ph
        following = relu(pre); following[:, 2], following[:, 3] = refill_scale[0]*lo, refill_scale[1]*hi
        background = pre+f[:, 6:7]
        available, floor = relu(background), relu(-background)
        spent = np.minimum(available, f[:, 6:7])
        terms = np.stack((transport-z, memory, query, phase, relay, input_, refill, floor, -spent), axis=1)
        flow = transport-z+memory+query+phase+relay+input_+refill+floor-spent
        return following, pre, flow, terms, available, spent

    def _scores(self, f, z, generated_g):
        """Three signed clips stay separate; their dispersion is not discarded."""
        k = self.k
        background = np.stack((f[:, 0:1]*k.readout_carry, f[:, 1:2]*k.readout_record,
            f[:, 2:3]*k.readout_query, f[:, 6:7]*k.readout_candidate,
            f[:, 7:8]*k.readout_relay, z@k.readout_capacity.T), axis=1).sum(1)
        common = generated_g@k.readout_common
        correction = (f[:, 3:6]-generated_g)@k.readout_E.T
        return background+common[:, None]-generated_g@k.debt_weights.T+correction

    def step(self, old, x):
        old = np.asarray(old, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
        f, old_z, wide, _ = self._read(old); k = self.k
        if x.shape != (len(old),) or not np.isfinite(x).all():
            raise ValueError('Expected one finite input per state')
        old_native = np.column_stack((f, old_z)); H, L = old[:, 1], old[:, 2]
        old_query = wide[:, 2]-wide[:, 0]+wide@k.old_query_residual
        echo = k.query_retention*old_query+wide@k.context_return+f[:, [1, 6, 7]]@k.old_event_echo
        q, d, qpre, dpre = pair(L/2, 2*(x-H))
        carry_pre = f[:, :2]@k.memory_pair+old_native@k.memory_history+k.memory_input*x
        m = relu(carry_pre)
        raw = wide[:, 3:6]@k.width_intrinsic.T+f[:, [0, 1, 2, 6, 7]]@k.width_controls.T+old_z@k.width_capacity.T
        width, offset, centre = raw.T
        delta = np.column_stack((centre-width/2, centre+width/2, -offset))
        g, cap_return = signed_caps(qpre, delta)
        widths = np.column_stack((g[:, 1]-g[:, 0], -g[:, 2], (g[:, 0]+g[:, 1])/2))
        a = k.input5_abs/k.candidate_scale*relu(old[:, 3]-x)
        relay_drive = old_native@k.relay_budget+k.relay_input*x
        b = relu(relu(relu(relay_drive)-k.relay_deficit*f[:, 2])-k.relay_self*f[:, 7])
        z, carrier_pre, flow, terms, available, spent = self._carriers(f, old_z, x)
        pre = np.empty_like(old)
        pre[:, P] = carrier_pre*k.candidate_cost
        pre[:, 2], pre[:, 4], pre[:, 7] = carry_pre*k.memory_scale, dpre*k.record_input_scale, qpre*k.comparator_scales[2]
        pre[:, [1, 6, 8]] = (qpre[:, None]-delta)*k.comparator_scales[[0, 1, 3]]
        pre[:, 5] = k.input5_abs*(old[:, 3]-x)
        pre[:, 9] = k.relay_scale*(relay_drive-k.relay_deficit*f[:, 2]-k.relay_self*f[:, 7])
        floors = relu(-pre[:, Q]); events = np.column_stack((d*k.record_input_scale, a*k.candidate_scale, b*k.relay_scale))
        reference = H+L/2; lower = reference-q
        signed_query = -lower+k.high_excess*relu(-qpre)-(cap_return*k.cap_gains).sum(1)+echo
        signed_query += (floors[:, J]*k.query_floor_gains).sum(1)
        level = (1-k.level_mix)*old[:, 0]+k.level_mix*x-echo+(floors*k.level_floor_gains).sum(1)+(events*k.level_event_gains).sum(1)
        packet = -signed_query; tau = k.inverse_price*(packet-level)
        fields = np.column_stack((m, d, q, widths, a, b)); innovation = fields-wide[:, :8]
        Lnew = L+(innovation*k.corridor_read[:8]).sum(1)+(flow*k.corridor_read[8:]).sum(1)
        residual = (innovation*k.midpoint_residual[:8]).sum(1)+(flow*k.midpoint_residual[8:]).sum(1)
        leading = H+d; continuation = m-(f[:, 0]+f[:, 1]); Hnew = leading+continuation+residual
        base_terms = np.column_stack(((k.base_alpha+k.base_beta)*f[:, 1], -k.base_beta*(leading-H),
            k.base_kq*(lower-reference+f[:, 2]), k.base_kappa*(a-f[:, 6]), k.base_alpha*continuation,
            ((widths-wide[:, 3:6])*k.base_phase).sum(1), k.base_relay*(b-f[:, 7]), (flow*k.base_capacity).sum(1)))
        c = old[:, 0]+base_terms.sum(1)
        new = np.column_stack((c, Hnew, Lnew, tau, widths[:, :2], a, b, z))
        actual, actual_z, _, rounding = self._read(new)
        scores = self._scores(actual, actual_z, g)
        if not np.isfinite(new).all() or not np.isfinite(scores).all():
            raise ValueError('Circuit overflow')
        return new, dict(scores=scores, pre=pre, physical=self._physical(actual, actual_z),
            c=c, H=Hnew, L=Lnew, tau=tau, a=a, b=b, g=g, delta=delta, packet=packet, level=level,
            available=available, spent=spent, carrier_flow=flow, carrier_terms=terms,
            base_terms=base_terms, recovery_adjustment=rounding)

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not 1 <= x.shape[1] <= 10 or not np.isfinite(x).all():
            raise ValueError('Expected finite inputs shaped (batch, steps), with 1 to 10 steps')
        state = self.zero(len(x)); states, rows = [], []
        for column in x.T:
            state, row = self.step(state, column); states.append(state); rows.append(row)
        trace = {key: np.stack([row[key] for row in rows], axis=1) for key in rows[0]}
        return np.stack(states, axis=1), trace['scores'], trace
