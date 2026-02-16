import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_confusion_matrix(
    cm,
    class_names,
    title,
    normalize=False,
    log_scale=False,
    output_path=None,
):
    """
    cm           : confusion matrix (numpy array)
    normalize    : if True, row-normalize (recall-focused)
    log_scale    : if True, use log-scaled colorbar
    output_path  : if set, saves figure as PDF
    """

    if normalize:
        cm = cm.astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))

    norm = LogNorm(vmin=cm[cm > 0].min(), vmax=cm.max()) if log_scale else None
    im = ax.imshow(cm, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion" if normalize else "Number of samples")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"

            # Decide text color based on background intensity
            if log_scale:
                threshold = (cm[cm > 0].min() * cm.max()) ** 0.5
            else:
                threshold = cm.max() / 2.0

            color = "white" if value < threshold else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=11,
                color=color,
            )

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ======================
# Training confusion matrix - Enter Confusion Matrix data here
# ======================
cm_training = np.array([[7278, 75], [7, 374]])

plot_confusion_matrix(
    cm_training,
    class_names=["Benign", "Malicious"],
    title="Training Confusion Matrix (Counts, Log Scale)",
    log_scale=True,
    # output_path="confusion_training_counts.pdf",
)

plot_confusion_matrix(
    cm_training,
    class_names=["Benign", "Malicious"],
    title="Training Confusion Matrix (Normalized)",
    normalize=True,
    # output_path="confusion_training_normalized.pdf",
)

# ======================
# Testing confusion matrix - Enter Confusion Matrix data here
# ======================
cm_testing = np.array([[3438, 65], [23, 158]])

plot_confusion_matrix(
    cm_testing,
    class_names=["Benign", "Malicious"],
    title="Testing Confusion Matrix (Counts, Log Scale)",
    log_scale=True,
    # output_path="confusion_testing_counts.pdf",
)

plot_confusion_matrix(
    cm_testing,
    class_names=["Benign", "Malicious"],
    title="Testing Confusion Matrix (Normalized)",
    normalize=True,
    # output_path="confusion_testing_normalized.pdf",
)
