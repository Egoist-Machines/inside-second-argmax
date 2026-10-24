# Inside a second-argmax network

A circuit account of ARC’s 432-parameter recurrent network. The article explains candidate writes, carrier supply and admission feedback through two interactive examples. Rational checks establish conditional source effects and a false-record error family.

## Read

```sh
python3 -m http.server 8901 --bind 127.0.0.1
```

Open [localhost:8901](http://127.0.0.1:8901/). No build step or external assets. [Pseudocode](pseudocode.html) · [Technical note](technical.html).

## Reproduce

Use Python 3.12 or newer. Node is needed only for the browser checks.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r reproduce/requirements.txt
python reproduce/check_circuit.py
python reproduce/proofs.py --check results/proofs.json
node --test reproduce/browser.spec.mjs
```

- [Circuit program](reproduce/circuit.py): sixteen stored coordinates, complete learned filters and all output scores.
- [Exact proofs](reproduce/proofs.py): supplied-write effects against every rival and the false-record family, including its true-record control.
- [Selected results](results/): pinned weights, derived tables, fixed examples, proof records and validation scope.

All 432 parameters are accounted for. Mechanistic claims are conditional; dense learned filters remain, and the account does not classify every error or derive task accuracy. A separate intervention study had an original event-label audit failure and an amended coverage miss. Both are recorded in [evidence.json](results/evidence.json); that larger study is not rerun by this kit.

## Citation

```bibtex
@misc{khachaturov2026secondargmax,
  author = {Khachaturov, David},
  title = {Inside a second-argmax network},
  year = {2026},
  howpublished = {Code and research note}
}
```

[BibTeX](CITATION.bib) · [Citation metadata](CITATION.cff)

Model and original analysis: [Alignment Research Center](https://github.com/alignment-research-center/alg-zoo) and Zihao Chen. David Khachaturov · [MIT-0](LICENSE) · [Sources](NOTICE.md).
