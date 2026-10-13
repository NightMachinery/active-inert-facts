# Supplementary artifacts

Files that accompany the paper. Everything here is plain text.

## `label-sets/`

The label sets from which new relations draw their labels (paper, Method,
"Label sets", and Appendix "Label sets").

- `labels/<name>.txt`: one label per line.
  - `animal.txt` is the animal label set of the main experiment, and the
    reference condition of the label-set comparison.
  - The other 13 files are the category label sets of the label-set
    comparison, one per WordNet 3.0 noun lexicographer file (`act.txt` is
    built from `noun.act`, and so on).
- `neighbor-pruning/<name>.tsv`: for each relation, the labels that
  neighbor pruning removes from that label set before the relation draws its
  labels. There is one row per removed label; a relation with no row loses
  nothing. The columns are:
  - `relation_id`: the relation's identifier;
  - `relation`: its display name, as printed in the paper;
  - `label`: the removed label;
  - `matched_string`: the relation's subject or answer string that removed it;
  - `hypernym_steps`: the WordNet hypernym distance between the label's
    primary sense and a noun sense of that string (0 to 4);
  - `kind`: `equal` (the label is the string), `same_sense` (they share a
    sense), `label_is_hyponym` or `label_is_hypernym` (the direction of the
    path).
- `manifest.json`: for every label set, its size, sha256, WordNet root synset,
  tokenizer rule, where it is used, and the number of relations neighbor
  pruning touches together with the fewest labels any relation keeps. It also
  records the thresholds shared by all sets: the `wordfreq` zipf floor, the
  near-duplicate cosine ceiling and the neighbor-pruning distance.

## `prompts/`

The prompts of the 70 relations of the main panel, as run (paper, Method and
Experimental setup). In every template, `{}` stands for the subject.

- `templates.tsv`: six templates per relation, one per row. The columns are:
  - `relation_id` and `relation`: as in `label-sets/`;
  - `role`: `pretrained` (the pretrained relation's query template),
    `fine-tuning` (the four new-relation fine-tuning templates) or
    `evaluation` (the new-relation evaluation template);
  - `template`: the template.
- `demonstrations.tsv`: the five demonstration subjects each relation reserves
  for each model, in the order they appear in the five-shot prompt. They are
  drawn per model, because the split is computed under each model's
  tokenizer, and they appear in no new-relation record. The columns are:
  - `relation_id` and `relation`: as above;
  - `model`: the model's Hugging Face id;
  - `subject`: the demonstration subject;
  - `answer`: the subject's own answer in the pretrained relation.

How a prompt is assembled:

- A new-relation fine-tuning or evaluation record is the template with the
  subject filled in, followed by a space and the label:
  `template.replace("{}", subject) + " " + label`. Training and scoring read
  the label's first token.
- The word `animal` in a new-relation template is the label set's category
  word. In the label-set comparison it is replaced by the name of the category
  label set (`act`, `artifact`, and so on; see `label-sets/labels/`).
- Pre-linearity reads the subject's last-token state in the `pretrained`
  template with the subject filled in, alone, with no demonstrations.
- Recall is the base model's candidate-restricted accuracy on that same
  prompt, read at its last token.
- The five-shot prompt is six lines joined by newlines: the five
  demonstrations, each `template.replace("{}", subject) + " " + answer` in
  file order, then the test line `template.replace("{}", test_subject)`, all
  under the `pretrained` template. No number in the paper is read from it.

## `relations/`

The pretrained relations as run, one file per model and relation (paper,
Method): `relations/<model>/<relation_id>.tsv`. `<model>` is the model's
Hugging Face id with `/` replaced by `__` (`openbmb/MiniCPM5-2B-Base` is
`openbmb__MiniCPM5-2B-Base`). The subjects, their split and the merging of
answers are computed under each model's tokenizer, so the files differ by
model. There is one row per subject the model ran, and the columns are:

- `subject`: the subject;
- `pretrained_answer`: the subject's answer in the source relation;
- `answer`: its effective answer. Answers whose first token is the same under
  the model's tokenizer are one answer, named by one of them, so `answer` can
  differ from `pretrained_answer`;
- `role`: `fine-tuning` (trained on, with the four fine-tuning templates, and
  scored on the evaluation template), `held-out` (never trained on; scored on
  all five templates) or `demonstration` (the five subjects of
  `prompts/demonstrations.tsv`, which appear in no new-relation record).

## `new-relations/`

The new relations, one file per model and relation:
`new-relations/<model>/<relation_id>.tsv`, with `<model>` as in `relations/`.
There is one row per answer, and the columns are:

- `answer`: an effective answer, as in `relations/`;
- `label_r0`: its label in the new relation of the main experiment;
- `label_r1`, `label_r2`: its label in the two re-randomized new relations,
  present only where the relation was repeated (every Hernandez and Town
  relation on every model except Qwen3.8-27B, and every BEAR relation on
  MiniCPM5-2B-Base). The repeats keep the subjects, roles and answers and draw
  new labels.

A subject's label is the label of its answer: join `relations/` and
`new-relations/` on `answer`. A `fine-tuning` or `held-out` subject's
new-relation record is the new-relation template with the subject filled in,
followed by a space and that label (see `prompts/`).

## `new-relations-label-sets/`

The new relations of the label-set comparison, which runs on
MiniCPM5-2B-Base only: `new-relations-label-sets/<category>/<relation_id>.tsv`
for each of the 13 category label sets of `label-sets/labels/`. The columns
are `answer` and `label` (drawn from `label-sets/labels/<category>.txt`). The
subjects, roles and answers are those of
`relations/openbmb__MiniCPM5-2B-Base/<relation_id>.tsv`, so join on `answer`
as above; the reference condition is that model's `label_r0` in
`new-relations/`.

## `models/`

- `leaderboard-tiny-2026-09-18.json5`: the Artificial Analysis ranking of tiny
  open-weight models on the day the models were chosen (paper, Experimental
  setup, "Models"): each entry's rank, name, organization, Intelligence Index,
  parameter count and context window, as shown on the page.

## `code/`

Our code. Keep `linroute/` and `paper/` side by side: `linroute/` reads the
registries in `paper/`.

- `linroute/`: the experiment code, which builds the new relations, fine-tunes
  the models, runs the probes and computes the analyses.
  - `linroute/`: the Python package;
  - `scripts/`: the command-line entry points;
  - `pyproject.toml`: install with `pip install -e code/linroute` (Python 3.12
    or later).
- `paper/`: the registries `linroute/` reads, at the paths it expects.
  - `metadata/`: the model, probe-name, relation-label and relation-set
    registries;
  - `plots/shared/`: the loaders of the relation-set and probe-name
    registries.

The scripts take the directories they read and write as arguments, except a
few analysis scripts, which read them from environment variables named
`LINROUTE_*`. Our run outputs (fine-tuned models, activation caches and result
rows) are not included; the scripts produce them.

## Licenses

The labels are lemmas of WordNet 3.0, copyright 2006 Princeton University,
used under the WordNet 3.0 license
(https://wordnet.princeton.edu/license-and-commercial-use).

The BEAR relations (every `relation_id` that starts with `bear_`) are adapted
from BEAR-big (Wiland, Ploner and Akbik, 2024, "BEAR: A Unified Framework for
Evaluating Relational Knowledge in Causal and Masked Language Models",
https://github.com/lm-pub-quiz/BEAR), which is released under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Adaptation means selection
and recounting rather than rewriting: one relation's rows are extracted from
the release, the single-value rule is applied, rows for subjects with no
published record are removed, and a display name is lower-cased where the
upstream casing would produce a slug our loaders refuse. No answer string is
edited and no row is repaired. What these files reproduce from BEAR-big is
distributed under CC BY-SA 4.0, with this attribution:

- in `prompts/templates.tsv`, the `pretrained` rows of the BEAR relations,
  except those of `bear_constellation` and `bear_taxon_rank`, whose template
  we wrote because BEAR-big ships none;
- in `prompts/demonstrations.tsv`, every row of a BEAR relation;
- in `label-sets/neighbor-pruning/`, the `matched_string` of every row of a
  BEAR relation, which is a BEAR-big subject or answer;
- in `relations/`, every file of a BEAR relation;
- in `new-relations/` and `new-relations-label-sets/`, the `answer` column of
  every file of a BEAR relation.

The Hernandez relations (`company_hq`, `landmark_in_country`,
`object_superclass`, `person_father`, `person_mother`,
`person_native_language`, `person_occupation`, `person_sport_position`,
`plays_pro_sport`, `product_by_company` and `star_constellation_name`) are
taken from the relation dataset of Hernandez et al. (2024,
"Linearity of Relation Decoding in Transformer Language Models",
https://github.com/evandez/relations), which is released under the MIT
License. What these files reproduce from it (their `pretrained` rows in
`prompts/templates.tsv`, their rows in `prompts/demonstrations.tsv`, their
`matched_string` values in `label-sets/neighbor-pruning/`, their files in
`relations/` and the `answer` column of their files in `new-relations/` and
`new-relations-label-sets/`) is distributed under that license, whose notice
is in `LICENSE-hernandez-relations.txt`.

The Town relations (`town_region` and every `relation_id` that starts
with `engineered_town_`) are ours, built from Wikidata.

The new-relation templates of every relation are ours.
