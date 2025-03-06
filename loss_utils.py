#########################   Instruction   ##########################
#####                   Used to classify tasks                  ####
####################################################################

import mlx.core as mx

def log_softmax(x, axis=1):
    x_max = mx.max(x, axis=axis, keepdims=True)
    log_sum_exp = mx.log(mx.sum(mx.exp(x - x_max), axis=axis, keepdims=True))
    return x - x_max - log_sum_exp


def cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss.
    Args:
        logits: Predicted logits of shape (batch_size, num_classes).
        labels: Ground truth labels of shape (batch_size,).
    Returns:
        Scalar loss value.
    """
    if logits.ndim != 2:
        raise ValueError(f"Logits must have shape (batch_size, num_classes), got {logits.shape}")
    if labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Labels must have shape (batch_size,), got {labels.shape}")

    num_classes = logits.shape[1]

    one_hot_labels = mx.eye(num_classes)[labels]

    log_probs = log_softmax(logits, axis=1)

    loss = -mx.sum(one_hot_labels * log_probs) / logits.shape[0]
    return loss