# Inside a second-argmax network

This repository traces how ARC’s 432-parameter recurrent network returns the second-largest input’s position. The article follows a successful record switch, then isolates how a candidate write changes positional evidence and how a false record causes a rank error. The two causal claims are verified with rational arithmetic under the conditions stated in the article.

## Read

[Article](https://egoistmachines.com/research/inside-second-argmax/) · [Pseudocode](https://egoistmachines.com/research/inside-second-argmax/pseudocode.html) · [Technical note](https://egoistmachines.com/research/inside-second-argmax/technical.html)

To serve a local copy:

```sh
python3 -m http.server 8901 --bind 127.0.0.1
```

Open [localhost:8901](http://127.0.0.1:8901/). No build step or external assets.

## Reproduce

Use Python 3.12 or newer. Node is needed only for the browser checks.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r reproduce/requirements.txt
python reproduce/check_circuit.py
python reproduce/proofs.py --check results/proofs.json
python reproduce/check_trace.py
python reproduce/evaluate_accuracy.py --check results/accuracy.json
node --test reproduce/browser.spec.mjs
```

- [Circuit program](reproduce/circuit.py): sixteen stored coordinates, complete learned filters and all output scores.
- [Exact proofs](reproduce/proofs.py): supplied-write effects against every rival and the false-record family, including its true-record control.
- [Successful trace](reproduce/check_trace.py) and [accuracy evaluation](reproduce/evaluate_accuracy.py): fixed native runs and a seeded 100,000-sequence estimate.
- [Selected results](results/): pinned weights, derived tables, fixed examples, proof records and validation scope.

The program includes every contribution from the 432 trained parameters. Some groups of weights still act as dense filters over input history, and explaining their roles, the remaining errors and overall accuracy requires further work. A larger intervention study first failed an event-label audit; a separate amended cohort passed the technical checks but missed a coverage target. Both outcomes are recorded in [evidence.json](results/evidence.json). Reproducing that larger study requires its separate implementation and reports.

## Citation

```bibtex
@misc{khachaturov2026secondargmax,
  author = {Khachaturov, David},
  title = {Inside a second-argmax network},
  year = {2026},
  date = {2026-10-26},
  howpublished = {Code and research note},
  url = {https://egoistmachines.com/research/inside-second-argmax/}
}
```

[BibTeX](CITATION.bib) · [Citation metadata](CITATION.cff) · [Issues and comments](https://github.com/Egoist-Machines/inside-second-argmax/issues)

Model and original analysis: [Alignment Research Center](https://github.com/alignment-research-center/alg-zoo) and Zihao Chen. David Khachaturov · [MIT-0](LICENSE) · [Sources](NOTICE.md).
