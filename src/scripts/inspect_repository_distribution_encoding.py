from __future__ import annotations

import argparse

from sastllm.classification.encoders import ClusterDistributionEncoder

from .inspection_common import (
    add_repository_selector,
    format_vector,
    initialize_script,
    load_labels,
    load_num_clusters,
    load_repository_inspection,
    print_nonzero_vector,
    print_repository_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one repository and its ClusterDistributionEncoder output.")
    add_repository_selector(parser)
    args = parser.parse_args()

    initialize_script()
    num_clusters = load_num_clusters(args.config, args.mode, args.num_clusters)
    labels = load_labels()
    inspection = load_repository_inspection(repository_id=args.repository_id, repository_name=args.repository_name)
    repository = inspection.to_classification_entity()

    print_repository_tree(inspection, max_description_chars=args.max_description_chars)
    print()
    print("ClusterDistributionEncoder")
    print(f"num_clusters={num_clusters}")
    print(f"raw_cluster_counts={repository.data or {}}")

    encoder = ClusterDistributionEncoder(num_clusters=num_clusters, labels=labels, matrix_normalization=False)
    encoding = encoder.encode([repository])
    vector = encoding.features[0]
    print(f"feature_shape={tuple(encoding.features.shape)}")
    print("nonzero_features:")
    print_nonzero_vector(vector)
    if args.show_full_vector:
        print(f"full_vector={format_vector(vector.tolist())}")


if __name__ == "__main__":
    main()
