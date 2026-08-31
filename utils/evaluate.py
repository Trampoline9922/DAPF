
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, roc_auc_score
from torch.nn.functional import softmax

from .logreg import LogReg


def sample_mean_std(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("values must not be empty")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, std


def _auc_score(labels: torch.Tensor, logits: torch.Tensor, nb_classes: int) -> float:
    probs = softmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    if nb_classes == 2:
        return float(roc_auc_score(y_true=y_true, y_score=probs[:, 1]))
    return float(roc_auc_score(y_true=y_true, y_score=probs, multi_class="ovr"))


def _evaluate_classifier_once(model, embs, labels, nb_classes):
    model.eval()
    with torch.no_grad():
        logits = model(embs)
        preds = torch.argmax(logits, dim=1)
        macro = float(f1_score(labels.cpu(), preds.cpu(), average="macro"))
        micro = float(f1_score(labels.cpu(), preds.cpu(), average="micro"))
        auc = _auc_score(labels, logits, nb_classes)
    return macro, micro, auc


def evaluate(
    embeds,
    ratio,
    idx_train,
    idx_val,
    idx_test,
    label,
    nb_classes,
    device,
    dataset,
    lr,
    wd,
    starttime,
    isTest=True,
    *,
    classifier_repeats=50,
    classifier_epochs=200,
    classifier_seed=0,
    write_result=True,
):
    if classifier_repeats <= 0 or classifier_epochs <= 0:
        raise ValueError("classifier_repeats and classifier_epochs must be positive")

    embeds = embeds.detach()
    train_embs = embeds[idx_train].detach()
    val_embs = embeds[idx_val].detach()
    test_embs = embeds[idx_test].detach()
    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)

    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()
    macro_f1s, micro_f1s, auc_scores = [], [], []

    for repeat in range(int(classifier_repeats)):
        seed = int(classifier_seed) + repeat
        # fork_rng prevents evaluation seeds from altering any later training RNG
        # state if evaluate() is reused inside larger experiment scripts.
        devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            clf = LogReg(hid_units, nb_classes).to(device)
            optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
            best_val_micro = -float("inf")
            best_state = None

            for _ in range(int(classifier_epochs)):
                clf.train()
                optimizer.zero_grad()
                logits = clf(train_embs)
                loss = xent(logits, train_lbls)
                loss.backward()
                optimizer.step()

                clf.eval()
                with torch.no_grad():
                    val_logits = clf(val_embs)
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_micro = float(
                        f1_score(val_lbls.cpu(), val_preds.cpu(), average="micro")
                    )
                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_state = copy.deepcopy(clf.state_dict())

            if best_state is None:
                raise RuntimeError("No classifier checkpoint was selected")
            clf.load_state_dict(best_state)
            macro, micro, auc = _evaluate_classifier_once(
                clf, test_embs, test_lbls, nb_classes
            )
            macro_f1s.append(macro)
            micro_f1s.append(micro)
            auc_scores.append(auc)

    macro_mean, macro_std = sample_mean_std(macro_f1s)
    micro_mean, micro_std = sample_mean_std(micro_f1s)
    auc_mean, auc_std = sample_mean_std(auc_scores)

    result = {
        "macro_f1": macro_mean * 100.0,
        "macro_f1_std": macro_std * 100.0,
        "micro_f1": micro_mean * 100.0,
        "micro_f1_std": micro_std * 100.0,
        "auc": auc_mean * 100.0,
        "auc_std": auc_std * 100.0,
        "classifier_repeats": int(classifier_repeats),
        "classifier_epochs": int(classifier_epochs),
        "classifier_seed": int(classifier_seed),
        "selection_metric": "validation_micro_f1",
        "macro_f1_values": [v * 100.0 for v in macro_f1s],
        "micro_f1_values": [v * 100.0 for v in micro_f1s],
        "auc_values": [v * 100.0 for v in auc_scores],
    }

    if isTest:
        print(
            "\t[Test Classification] "
            f"Macro-F1: {result['macro_f1']:.2f} ± {result['macro_f1_std']:.2f}  "
            f"Micro-F1: {result['micro_f1']:.2f} ± {result['micro_f1_std']:.2f}  "
            f"AUC: {result['auc']:.2f} ± {result['auc_std']:.2f}"
        )

    if write_result:
        with open(f"result_{dataset}{ratio}.txt", "a", encoding="utf-8") as f:
            stamp = starttime.strftime("%Y-%m-%d %H:%M") if starttime is not None else "NA"
            f.write(
                f"{stamp}\t"
                f"Ma-F1_mean: {result['macro_f1']:.2f} +/- {result['macro_f1_std']:.2f}\t"
                f"Mi-F1_mean: {result['micro_f1']:.2f} +/- {result['micro_f1_std']:.2f}\t"
                f"AUC_mean: {result['auc']:.2f} +/- {result['auc_std']:.2f}\n"
            )

    if not isTest:
        # Kept for compatibility with code that used the former return path.
        return result["macro_f1"], result["micro_f1"]
    return result


def run_kmeans(
    x,
    y,
    k,
    starttime,
    dataset,
    *,
    seeds=tuple(range(10)),
    n_init=10,
    normalize_embeddings=False,
    write_result=True,
):

    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one clustering seed is required")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    x_tensor = x.detach().cpu().float() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
    if normalize_embeddings:
        x_tensor = torch.nn.functional.normalize(x_tensor, p=2, dim=1)
    x_np = x_tensor.numpy()
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    nmi_values, ari_values = [], []
    for seed in seeds:
        estimator = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=int(n_init),
            random_state=seed,
        )
        y_pred = estimator.fit_predict(x_np)
        nmi_values.append(
            float(normalized_mutual_info_score(y_np, y_pred, average_method="arithmetic"))
        )
        ari_values.append(float(adjusted_rand_score(y_np, y_pred)))

    nmi_mean, nmi_std = sample_mean_std(nmi_values)
    ari_mean, ari_std = sample_mean_std(ari_values)
    result = {
        "nmi": nmi_mean * 100.0,
        "nmi_std": nmi_std * 100.0,
        "ari": ari_mean * 100.0,
        "ari_std": ari_std * 100.0,
        "nmi_values": [v * 100.0 for v in nmi_values],
        "ari_values": [v * 100.0 for v in ari_values],
        "seeds": seeds,
        "n_init": int(n_init),
        "normalize_embeddings": bool(normalize_embeddings),
    }

