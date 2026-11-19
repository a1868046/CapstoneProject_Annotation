# ATC Inter-Annotator Agreement Calculator

Compares intent annotations from two annotators. V8 is the latest supplied version; earlier scripts are retained for traceability. All scripts use Python 3 standard-library modules only.

## Run V8

Export two Label Studio JSON files containing independently submitted annotations. Each task needs `data.text` and a stable identifier (`sample_id`, `source_file` plus `turn_id`, or `turn_id`). Each task must contain exactly one non-cancelled submitted annotation. Intent regions use `from_name: intent`, a single label, and valid offsets. Matched transcripts must have identical text. Predictions are not used.

```sh
python3 IAA_Calculator_v8.py annotator_a.json annotator_b.json --output-dir results
```

Only shared task IDs are scored; check the printed input and matching counts. Draft mode (`--source draft`) is diagnostic only. Independence is not verified by the program.

## Outputs

- `agreement_report.csv`: counts, observed/chance agreement and Cohen’s kappa.
- `disagreements.csv`: differing token label pairs.
- `validation_flags.csv`: annotation structure issues; review before interpreting scores.
- `boundary_differences.csv`: unmatched whitespace-trimmed spans.
- `relationship_differences.csv`: unmatched labelled relationships.

V8 masks formatting markers and comments, retains event markers including `[speaker]`, and pairs distinct overlapping labels per token, matching common labels first and padding with `UNLABELLED`. Kappa is calculated over these constructed pairs, not independent span units. Scores from V2–V5 (layer-based) are not directly comparable to V6–V8. Unmatched labels are paired in sorted order, which is a methodological choice. Do not publish real transcripts or generated reports without client approval.

## Validation

All eight scripts successfully displayed CLI help. V8 was smoke-tested with synthetic JSON containing one differing intent label and one unlabelled word; it produced all five CSV outputs. This is not full validation on client data. The original CSV script returns 1.0 for a degenerate all-agree single-category case; later versions report undefined kappa in that case.

See [CHANGELOG.md](CHANGELOG.md) for version history. Future changes should use descriptive commits and pull requests.
