from __future__ import annotations

import argparse

from sastllm.classification.encoders import OrderedFunctionalityTimeSeriesEncoder

from .inspection_common import (
    add_repository_selector,
    format_vector,
    initialize_script,
    load_labels,
    load_num_clusters,
    load_repository_inspection,
    print_repository_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one repository and its OrderedFunctionalityTimeSeriesEncoder output.")
    add_repository_selector(parser)
    parser.add_argument("--max-sequence-length", type=int, help="Optional sequence length used for padding/truncation.")
    parser.add_argument("--truncation", choices=("first", "last"), default="first", help="Which side of long sequences to keep.")
    args = parser.parse_args()

    initialize_script()
    num_clusters = load_num_clusters(args.config, args.mode, args.num_clusters)
    labels = load_labels()
    inspection = load_repository_inspection(repository_id=args.repository_id, repository_name=args.repository_name)
    repository = inspection.to_classification_entity()

    print_repository_tree(inspection, max_description_chars=args.max_description_chars)
    print()
    print("OrderedFunctionalityTimeSeriesEncoder")
    print(f"num_clusters={num_clusters}")
    print(f"max_sequence_length={args.max_sequence_length}")
    print(f"truncation={args.truncation}")

    encoder = OrderedFunctionalityTimeSeriesEncoder(
        num_clusters=num_clusters,
        labels=labels,
        max_sequence_length=args.max_sequence_length,
        truncation=args.truncation,
    )
    encoding = encoder.encode([repository])
    matrix = encoding.features[0]
    sequence_length = int(encoding.sequence_lengths[0]) if encoding.sequence_lengths is not None else matrix.shape[0]
    ordered = sorted(
        (functionality for functionality in inspection.functionalities if functionality.cluster_id is not None),
        key=lambda functionality: functionality.functionality_id,
    )
    if args.max_sequence_length is not None and len(ordered) > args.max_sequence_length:
        ordered = ordered[: args.max_sequence_length] if args.truncation == "first" else ordered[-args.max_sequence_length :]

    print(f"feature_shape={tuple(encoding.features.shape)}")
    print(f"sequence_length={sequence_length}")
    print("timesteps:")
    if not ordered:
        print("  (no clustered functionalities)")
    for step, functionality in enumerate(ordered):
        row = matrix[step]
        active_columns = row.nonzero()[0].tolist()
        active = active_columns[0] if active_columns else None
        print(f"  t={step} functionality_id={functionality.functionality_id} lines={functionality.start_line}-{functionality.end_line} cluster_id={functionality.cluster_id} active_column={active}")
        if args.show_full_vector:
            print(f"    row={format_vector(row.tolist())}")

    if matrix.shape[0] > sequence_length:
        print(f"padding_rows={matrix.shape[0] - sequence_length}")


if __name__ == "__main__":
    main()
