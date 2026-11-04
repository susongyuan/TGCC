import torch


def task_metrics(logits, labels, mask):
    pred = logits.argmax(-1)
    p = pred[mask]
    y = labels[mask]
    probs = logits.softmax(-1)[mask]
    if not len(y):
        return {"count": 0}
    f1 = []
    for k in range(logits.shape[-1]):
        tp = ((p == k) & (y == k)).sum().item()
        fp = ((p == k) & (y != k)).sum().item()
        fn = ((p != k) & (y == k)).sum().item()
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    conf = probs.max(-1).values
    correct = (p == y).float()
    ece = 0.0
    for a in torch.linspace(0, 1, 11)[:-1]:
        m = (conf >= a) & (conf < a + 0.1)
        if m.any():
            ece += float(m.float().mean() * (conf[m].mean() - correct[m].mean()).abs())
    return {
        "count": len(y),
        "accuracy": float(correct.mean()),
        "macro_f1": sum(f1) / len(f1),
        "false_believed_done_rate": float(
            ((p == 2) & (y != 2)).sum() / max(1, (p == 2).sum())
        ),
        "ece10": ece,
    }
