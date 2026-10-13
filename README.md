# Knowledge That Is Linear Before Fine-Tuning Generalizes After It

Data and code for the paper by Faridoun Mehri and Hinrich Schütze.

This repository is meant to let a reader rerun our experiments. It holds every
pretrained relation, new relation and label set as run, the prompt templates and
our code: building the new relations, fine-tuning, the probes and every
analysis. Everything is in [`artifacts_v1/`](artifacts_v1/),
whose [README](artifacts_v1/README.md) documents every file and column:

- [`relations/`](artifacts_v1/relations/): the pretrained relations as run, one
  file per model and relation;
- [`new-relations/`](artifacts_v1/new-relations/): the new relations, with the
  labels of the main experiment and of the two re-randomized runs;
- [`new-relations-label-sets/`](artifacts_v1/new-relations-label-sets/): the new
  relations of the label-set comparison;
- [`label-sets/`](artifacts_v1/label-sets/): the label sets, and the labels that
  neighbor pruning removes for each relation;
- [`prompts/`](artifacts_v1/prompts/): every prompt template and the
  demonstration subjects;
- [`models/`](artifacts_v1/models/): the leaderboard behind the choice of
  models;
- [`code/`](artifacts_v1/code/): the experiment code.

All eleven models are open-weight. The paper gives the fine-tuning recipe, the
probe and its cross-validation, the label-set algorithm, the Town relations and
how every confidence interval is computed.

Our code ([`artifacts_v1/code/`](artifacts_v1/code/)) is released under the
[MIT license](artifacts_v1/code/LICENSE), and our data under
[CC BY-SA 4.0](LICENSE). Third-party material keeps its own terms: the BEAR
relations are adapted from BEAR-big and keep its CC BY-SA 4.0 license; the
Hernandez relations keep their MIT license
([notice](artifacts_v1/LICENSE-hernandez-relations.txt)); the labels are WordNet
3.0 lemmas; and the leaderboard in [`models/`](artifacts_v1/models/) records
Artificial Analysis's public ranking, which is not ours to license. See
[Licenses](artifacts_v1/README.md#licenses) for the details.
