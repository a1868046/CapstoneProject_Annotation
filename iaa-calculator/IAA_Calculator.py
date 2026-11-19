#!/usr/bin/env python3
"""Compare two independent annotators' categorical labels on identical units."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_csv(path, id_column, label_column):
    with open(path, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or id_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f'{path}: required columns are {id_column!r} and {label_column!r}')
        result = {}
        for line, row in enumerate(reader, 2):
            unit = (row[id_column] or '').strip()
            label = (row[label_column] or '').strip()
            if not unit or not label:
                raise ValueError(f'{path}, line {line}: empty ID or label')
            if unit in result:
                raise ValueError(f'{path}, line {line}: duplicate ID {unit!r}')
            result[unit] = label
        if not result:
            raise ValueError(f'{path}: no annotations found')
        return result


def evaluate(a, b):
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        raise ValueError(f'Annotation units do not match. Only A: {only_a[:10]}; only B: {only_b[:10]}')
    ids = sorted(a)
    n = len(ids)
    agree = sum(a[i] == b[i] for i in ids)
    observed = agree / n
    counts_a = Counter(a.values())
    counts_b = Counter(b.values())
    expected = sum(counts_a[label] * counts_b[label] for label in set(counts_a) | set(counts_b)) / (n*n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else None)
    disagreements = [{'unit_id': i, 'annotator_a': a[i], 'annotator_b': b[i]} for i in ids if a[i] != b[i]]
    return {'total_units': n, 'agreements': agree, 'disagreements': len(disagreements),
            'percent_agreement': round(observed * 100, 4),
            'expected_chance_agreement': round(expected, 6),
            'cohens_kappa': round(kappa, 6) if kappa is not None else None}, disagreements


def main():
    parser = argparse.ArgumentParser(description='Calculate inter-annotator agreement for categorical labels')
    parser.add_argument('annotator_a', type=Path)
    parser.add_argument('annotator_b', type=Path)
    parser.add_argument('--id-column', default='unit_id')
    parser.add_argument('--label-column', default='intent_label')
    parser.add_argument('--output-dir', type=Path, default=Path('iaa_results'))
    args = parser.parse_args()
    try:
        a = load_csv(args.annotator_a, args.id_column, args.label_column)
        b = load_csv(args.annotator_b, args.id_column, args.label_column)
        report, disagreements = evaluate(a, b)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'agreement_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        with (args.output_dir / 'disagreements.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['unit_id', 'annotator_a', 'annotator_b'])
            writer.writeheader()
            writer.writerows(disagreements)
        print(json.dumps(report, indent=2))
        print(f'\nSaved outputs to: {args.output_dir.resolve()}')
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
