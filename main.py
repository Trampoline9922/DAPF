import csv
import datetime
import importlib.metadata
import json
import math
import os
import pickle as pkl
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from module.DAPF import DAPF
from module.preprocess import normalize_adj, pathsim
from utils.evaluate import evaluate, run_kmeans
from utils.load_data import load_data
from utils.params import set_params

warnings.filterwarnings('ignore')


_BYTES_PER_MB = 1024.0 ** 2


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_mean_std(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, std


def summarize_epoch_timing(epoch_times):
    """Summarize full-run and post-warm-up epoch wall-clock timings.

    The first training epoch is excluded from the steady-state statistic because
    it can include one-off CUDA/library warm-up and cache initialization costs.
    When only one epoch is observed, steady-state values are reported as None
    instead of duplicating the warm-up-contaminated value.
    """
    epoch_times = [float(value) for value in epoch_times]
    avg_all, std_all = _sample_mean_std(epoch_times)

    steady_state_times = epoch_times[1:]
    if steady_state_times:
        steady_avg, steady_std = _sample_mean_std(steady_state_times)
    else:
        steady_avg, steady_std = None, None

    return {
        "avg_training_time_per_epoch_s": avg_all,
        "std_training_time_per_epoch_s": std_all,
        "steady_state_avg_epoch_time_s": steady_avg,
        "steady_state_std_epoch_time_s": steady_std,
        "steady_state_epochs_observed": len(steady_state_times),
    }


def _copy_sampling_stats(stats):
    return {name: dict(values) for name, values in stats.items()}


def summarize_path_sampling_history(history, path_budget):
    """Aggregate per-epoch sampling diagnostics without averaging ratios naively.

    Candidate paths are the schema-valid paths produced *after* the hop-level
    ``max_neighbors`` sampling and *before* the per-node ``path_budget`` cap.
    Counts are aggregated over target-node observations across epochs, so the
    reported ratios remain correct even if a future experiment changes the
    number of sampled target nodes from one epoch to another.
    """
    view_names = sorted({name for epoch_stats in history for name in epoch_stats})
    view_summaries = {}

    count_fields = (
        "num_start_nodes",
        "candidate_paths",
        "retained_paths",
        "zero_candidate_nodes",
        "insufficient_candidate_nodes",
        "nonzero_under_budget_nodes",
        "budget_saturated_nodes",
    )

    for view_name in view_names:
        records = [epoch_stats[view_name] for epoch_stats in history if view_name in epoch_stats]
        totals = {field: sum(int(record.get(field, 0)) for record in records) for field in count_fields}
        starts = totals["num_start_nodes"]
        max_retained = max((int(record.get("max_retained_per_node", 0)) for record in records), default=0)
        cap_applied = any(bool(record.get("global_cap_applied", False)) for record in records)
        candidate_definition = next(
            (record.get("candidate_definition") for record in records if record.get("candidate_definition")),
            "schema-valid paths after hop-level max_neighbors sampling and before path_budget retention",
        )

        view_summaries[view_name] = {
            "epochs_observed": len(records),
            "num_start_node_observations": starts,
            "candidate_paths_total": totals["candidate_paths"],
            "retained_paths_total": totals["retained_paths"],
            "avg_candidate_paths_per_node": totals["candidate_paths"] / starts if starts else 0.0,
            "avg_retained_paths_per_node": totals["retained_paths"] / starts if starts else 0.0,
            "zero_candidate_nodes": totals["zero_candidate_nodes"],
            "zero_candidate_node_ratio": totals["zero_candidate_nodes"] / starts if starts else 0.0,
            "insufficient_candidate_nodes": totals["insufficient_candidate_nodes"],
            "insufficient_candidate_node_ratio": totals["insufficient_candidate_nodes"] / starts if starts else 0.0,
            "nonzero_under_budget_nodes": totals["nonzero_under_budget_nodes"],
            "nonzero_under_budget_node_ratio": totals["nonzero_under_budget_nodes"] / starts if starts else 0.0,
            "budget_saturated_nodes": totals["budget_saturated_nodes"],
            "budget_saturation_ratio": totals["budget_saturated_nodes"] / starts if starts else 0.0,
            "max_retained_per_node_observed": max_retained,
            "global_cap_applied_any": cap_applied,
            "path_budget": path_budget,
            "candidate_definition": candidate_definition,
        }

    overall_starts = sum(v["num_start_node_observations"] for v in view_summaries.values())
    overall_candidates = sum(v["candidate_paths_total"] for v in view_summaries.values())
    overall_retained = sum(v["retained_paths_total"] for v in view_summaries.values())
    overall_zero = sum(v["zero_candidate_nodes"] for v in view_summaries.values())
    overall_insufficient = sum(v["insufficient_candidate_nodes"] for v in view_summaries.values())
    overall_nonzero_under = sum(v["nonzero_under_budget_nodes"] for v in view_summaries.values())
    overall_saturated = sum(v["budget_saturated_nodes"] for v in view_summaries.values())

    overall = {
        "num_start_node_observations": overall_starts,
        "avg_candidate_paths_per_node": overall_candidates / overall_starts if overall_starts else 0.0,
        "avg_retained_paths_per_node": overall_retained / overall_starts if overall_starts else 0.0,
        "zero_candidate_node_ratio": overall_zero / overall_starts if overall_starts else 0.0,
        "insufficient_candidate_node_ratio": overall_insufficient / overall_starts if overall_starts else 0.0,
        "nonzero_under_budget_node_ratio": overall_nonzero_under / overall_starts if overall_starts else 0.0,
        "budget_saturation_ratio": overall_saturated / overall_starts if overall_starts else 0.0,
        "path_budget": path_budget,
        "candidate_definition": (
            "schema-valid paths after hop-level max_neighbors sampling and before path_budget retention"
        ),
    }
    return {"overall": overall, "views": view_summaries}


def _summarize_stage_profiles(profile_history):
    if not profile_history:
        return {}
    keys = sorted({key for profile in profile_history for key in profile})
    summary = {}
    for key in keys:
        values = [float(profile.get(key, 0.0)) for profile in profile_history]
        mean, std = _sample_mean_std(values)
        summary[key] = {"mean_s": mean, "std_s": std, "values_s": values}
    return summary


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_metadata(device):
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": _package_version("scikit-learn"),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        metadata["gpu_total_memory_mb"] = props.total_memory / _BYTES_PER_MB
    else:
        metadata["gpu_name"] = None
        metadata["gpu_total_memory_mb"] = None
    return metadata


def _append_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def save_experiment_statistics(args, performance, output_dir=None):
    """Persist one complete run plus reviewer-facing CSV summaries."""
    output_dir = Path(output_dir if output_dir is not None else args.experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = performance.get("run_timestamp") or datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    file_stamp = timestamp.replace(":", "").replace("-", "")
    tag = (getattr(args, "experiment_tag", None) or "run").strip().replace(" ", "_")
    run_id = f"{args.dataset}_{tag}_K{args.path_budget}_seed{args.seed}_{file_stamp}"

    config_fields = [
        "dataset", "seed", "path_budget", "max_neighbors", "path_sample_ratio",
        "max_total_paths", "num_cluster", "tau", "lam_proto", "negative_mode",
        "use_positional_encoding", "prototype_update_interval", "prototype_batch_size",
        "prototype_kmeans_iters", "prototype_kmeans_restarts", "prototype_seed",
        "classifier_repeats", "classifier_epochs", "classifier_seed",
        "clustering_n_init", "clustering_seeds", "cluster_normalize", "nb_epochs",
        "patience", "lr", "l2_coef", "dropout", "experiment_tag",
    ]
    config = {field: getattr(args, field, None) for field in config_fields}

    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
        "config": config,
        "performance": performance,
    }
    json_path = output_dir / f"{run_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    timing = performance.get("timing", {})
    memory = performance.get("memory", {})
    clustering = performance.get("clustering", {})
    overall_sampling = performance.get("path_sampling_summary", {}).get("overall", {})
    parameter_count = performance.get("parameter_count", {})

    run_summary_fields = [
        "run_id", "timestamp", "experiment_tag", "dataset", "seed", "path_budget",
        "max_neighbors", "path_sample_ratio", "max_total_paths", "train_ratio",
        "macro_f1", "macro_f1_std", "micro_f1", "micro_f1_std", "auc", "auc_std",
        "nmi", "nmi_std", "ari", "ari_std", "epochs_ran", "best_train_loss",
        "avg_training_time_per_epoch_s", "std_training_time_per_epoch_s",
        "steady_state_avg_epoch_time_s", "steady_state_std_epoch_time_s",
        "steady_state_epochs_observed",
        "preprocessing_time_s", "inference_time_s", "evaluation_time_s",
        "end_to_end_time_s", "peak_gpu_memory_mb", "peak_gpu_reserved_mb",
        "avg_candidate_paths_per_node", "avg_retained_paths_per_node",
        "zero_candidate_node_ratio", "insufficient_candidate_node_ratio",
        "budget_saturation_ratio", "trainable_parameters", "total_parameters",
    ]
    run_rows = []
    for ratio, cls in sorted(performance.get("classification", {}).items(), key=lambda item: int(item[0])):
        run_rows.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_tag": getattr(args, "experiment_tag", ""),
            "dataset": args.dataset,
            "seed": args.seed,
            "path_budget": args.path_budget,
            "max_neighbors": args.max_neighbors,
            "path_sample_ratio": args.path_sample_ratio,
            "max_total_paths": args.max_total_paths,
            "train_ratio": int(ratio),
            "macro_f1": cls.get("macro_f1"),
            "macro_f1_std": cls.get("macro_f1_std"),
            "micro_f1": cls.get("micro_f1"),
            "micro_f1_std": cls.get("micro_f1_std"),
            "auc": cls.get("auc"),
            "auc_std": cls.get("auc_std"),
            "nmi": clustering.get("nmi"),
            "nmi_std": clustering.get("nmi_std"),
            "ari": clustering.get("ari"),
            "ari_std": clustering.get("ari_std"),
            "epochs_ran": performance.get("epochs_ran"),
            "best_train_loss": performance.get("best_train_loss"),
            "avg_training_time_per_epoch_s": timing.get("avg_training_time_per_epoch_s"),
            "std_training_time_per_epoch_s": timing.get("std_training_time_per_epoch_s"),
            "steady_state_avg_epoch_time_s": timing.get("steady_state_avg_epoch_time_s"),
            "steady_state_std_epoch_time_s": timing.get("steady_state_std_epoch_time_s"),
            "steady_state_epochs_observed": timing.get("steady_state_epochs_observed"),
            "preprocessing_time_s": timing.get("preprocessing_time_s"),
            "inference_time_s": timing.get("inference_time_s"),
            "evaluation_time_s": timing.get("evaluation_time_s"),
            "end_to_end_time_s": timing.get("end_to_end_time_s"),
            "peak_gpu_memory_mb": memory.get("peak_gpu_memory_mb"),
            "peak_gpu_reserved_mb": memory.get("peak_gpu_reserved_mb"),
            "avg_candidate_paths_per_node": overall_sampling.get("avg_candidate_paths_per_node"),
            "avg_retained_paths_per_node": overall_sampling.get("avg_retained_paths_per_node"),
            "zero_candidate_node_ratio": overall_sampling.get("zero_candidate_node_ratio"),
            "insufficient_candidate_node_ratio": overall_sampling.get("insufficient_candidate_node_ratio"),
            "budget_saturation_ratio": overall_sampling.get("budget_saturation_ratio"),
            "trainable_parameters": parameter_count.get("trainable"),
            "total_parameters": parameter_count.get("total"),
        })

    run_summary_csv = output_dir / "path_budget_run_summary.csv"
    _append_csv(run_summary_csv, run_summary_fields, run_rows)

    sampling_fields = [
        "run_id", "timestamp", "experiment_tag", "dataset", "seed", "path_budget",
        "max_neighbors", "meta_path", "epochs_observed", "num_start_node_observations",
        "avg_candidate_paths_per_node", "avg_retained_paths_per_node",
        "zero_candidate_node_ratio", "insufficient_candidate_node_ratio",
        "nonzero_under_budget_node_ratio", "budget_saturation_ratio",
        "max_retained_per_node_observed", "global_cap_applied_any",
    ]
    sampling_rows = []
    for meta_path, stats in sorted(performance.get("path_sampling_summary", {}).get("views", {}).items()):
        sampling_rows.append({
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_tag": getattr(args, "experiment_tag", ""),
            "dataset": args.dataset,
            "seed": args.seed,
            "path_budget": args.path_budget,
            "max_neighbors": args.max_neighbors,
            "meta_path": meta_path,
            **stats,
        })
    sampling_csv = output_dir / "path_sampling_summary.csv"
    _append_csv(sampling_csv, sampling_fields, sampling_rows)

    return {
        "json": json_path,
        "run_summary_csv": run_summary_csv,
        "sampling_csv": sampling_csv,
    }


def train(return_performance=False):
    args = set_params()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    seed_everything(args.seed)

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    end_to_end_started = time.perf_counter()

    data_loading_started = time.perf_counter()
    loaded = load_data(args.dataset, args.ratio, args.type_num)
    data_loading_seconds = time.perf_counter() - data_loading_started
    if len(loaded) == 8:
        nei_index, feats, adjs, label, idx_train, idx_val, idx_test, path_relations = loaded
    else:
        nei_index, feats, adjs, label, idx_train, idx_val, idx_test = loaded
        path_relations = None

    nb_classes = label.shape[-1]
    if args.dataset == 'aminer':
        feats_dim_list = [64, 64, 64]
    else:
        feats_dim_list = [feat.shape[1] for feat in feats]
    sub_num = len(adjs)

    print("Dataset:", args.dataset)
    print("The number of meta-paths:", sub_num)
    print("The dim of different kinds' nodes' feature:", feats_dim_list)

    preprocessing_started = time.perf_counter()
    # Structural meta-path graphs used by AS-View. Feature/edge masking from
    # the legacy code is intentionally not applied because it is not part of
    # the DAPF objective described in the manuscript.
    adjs = pathsim(adjs, args.nei_max)
    adjs_norm = [normalize_adj(adj) for adj in adjs]
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    model_setup_started = time.perf_counter()
    model = DAPF(
        feats_dim_list,
        sub_num,
        args.hidden_dim,
        args.embed_dim,
        args.tau,
        adjs_norm,
        args.lam_proto,
        args.dropout,
        nei_index,
        args.dataset,
        offline_path_file=None,
        path_budget=args.path_budget,
        max_neighbors=args.max_neighbors,
        use_positional_encoding=args.use_positional_encoding,
        path_sample_ratio=args.path_sample_ratio,
        max_total_paths=args.max_total_paths,
        path_relations=path_relations,
        negative_mode=args.negative_mode,
        prototype_update_interval=args.prototype_update_interval,
        prototype_batch_size=args.prototype_batch_size,
        prototype_kmeans_iters=args.prototype_kmeans_iters,
        prototype_kmeans_restarts=args.prototype_kmeans_restarts,
        prototype_seed=args.prototype_seed,
        profile_training=args.profile_training,
    ).to(device)

    feats = [feat.to(device) for feat in feats]
    label = label.to(device)
    idx_train = [idx.to(device) for idx in idx_train]
    idx_val = [idx.to(device) for idx in idx_val]
    idx_test = [idx.to(device) for idx in idx_test]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    _sync_device(device)
    model_setup_seconds = time.perf_counter() - model_setup_started

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_loss = float('inf')
    cnt_wait = 0
    starttime = datetime.datetime.now()
    epochs_ran = 0
    epoch_times = []
    path_sampling_history = []
    profile_history = []
    inference_seconds = 0.0
    final_sampling_stats = {}
    peak_inference_gpu_memory_mb = None

    if args.save_emb:
        for epoch in range(args.nb_epochs):
            print("Epoch:", epoch)
            _sync_device(device)
            epoch_started = time.perf_counter()

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(feats, num_cluster=args.num_cluster)

            if args.profile_training and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_started = time.perf_counter() if args.profile_training else None
            loss.backward()
            if args.profile_training:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model.last_profile["backward"] = time.perf_counter() - backward_started

            optimizer.step()
            _sync_device(device)
            epoch_times.append(time.perf_counter() - epoch_started)
            path_sampling_history.append(_copy_sampling_stats(model.path_sampling_stats))
            if args.profile_training:
                profile_history.append(dict(model.last_profile))
            epochs_ran = epoch + 1

            loss_value = float(loss.detach().cpu())
            print(f"total loss: {loss_value:.6f}; best: {best_loss:.6f}")
            diagnostics = model.last_diagnostics
            print(
                "loss_info: {loss_info:.6f}; loss_proto: {loss_proto:.6f}; "
                "prototype temperature [min/mean/max]: "
                "{prototype_temperature_min:.6g}/{prototype_temperature_mean:.6g}/"
                "{prototype_temperature_max:.6g}".format(**diagnostics)
            )
            if args.profile_training:
                profile_order = [
                    "path_sampling",
                    "path_encoding",
                    "as_view",
                    "semantic_fusion_projection",
                    "prototype_clustering",
                    "contrastive_losses",
                    "backward",
                ]
                print(
                    "profile (s): "
                    + "; ".join(
                        f"{name}={model.last_profile.get(name, 0.0):.4f}"
                        for name in profile_order
                    )
                )
            if loss_value < best_loss:
                best_loss = loss_value
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience:
                print('Early stopping!')
                break

        training_peak_gpu_memory_mb = None
        training_peak_gpu_reserved_mb = None
        if device.type == "cuda":
            training_peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB
            training_peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / _BYTES_PER_MB
            torch.cuda.reset_peak_memory_stats(device)

        # Recompute the representation after the last optimizer update. This
        # avoids saving the pre-update tensor cached during the final forward.
        model.eval()
        _sync_device(device)
        inference_started = time.perf_counter()
        with torch.no_grad():
            embeds = model.encode_embeddings(feats).detach()
        _sync_device(device)
        inference_seconds = time.perf_counter() - inference_started
        final_sampling_stats = _copy_sampling_stats(model.path_sampling_stats)
        if device.type == "cuda":
            peak_inference_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB

        os.makedirs(os.path.join("embeds", args.dataset), exist_ok=True)
        with open(os.path.join("embeds", args.dataset, f"{args.turn}.pkl"), "wb") as f:
            pkl.dump(embeds.cpu().numpy(), f)
        print("Embedding save finished.")
    else:
        training_peak_gpu_memory_mb = None
        training_peak_gpu_reserved_mb = None
        with open(os.path.join("embeds", args.dataset, f"{args.turn}.pkl"), "rb") as f:
            embeds = torch.from_numpy(pkl.load(f)).to(device)

    _sync_device(device)
    evaluation_started = time.perf_counter()
    clustering_result = run_kmeans(
        embeds,
        torch.argmax(label, dim=-1),
        nb_classes,
        starttime,
        args.dataset,
        seeds=args.clustering_seeds,
        n_init=args.clustering_n_init,
        normalize_embeddings=args.cluster_normalize,
    )

    classification_results = {}
    for split_idx, ratio in enumerate(args.ratio):
        classification_results[int(ratio)] = evaluate(
            embeds,
            ratio,
            idx_train[split_idx],
            idx_val[split_idx],
            idx_test[split_idx],
            label,
            nb_classes,
            device,
            args.dataset,
            args.eva_lr,
            args.eva_wd,
            starttime,
            classifier_repeats=args.classifier_repeats,
            classifier_epochs=args.classifier_epochs,
            classifier_seed=args.classifier_seed,
        )
    _sync_device(device)
    evaluation_seconds = time.perf_counter() - evaluation_started

    total_seconds = (datetime.datetime.now() - starttime).total_seconds()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    print(f"Total time: {total_seconds:.2f} s")

    epoch_timing_summary = summarize_epoch_timing(epoch_times)
    avg_epoch_seconds = epoch_timing_summary["avg_training_time_per_epoch_s"]
    std_epoch_seconds = epoch_timing_summary["std_training_time_per_epoch_s"]
    steady_avg_epoch_seconds = epoch_timing_summary["steady_state_avg_epoch_time_s"]
    steady_std_epoch_seconds = epoch_timing_summary["steady_state_std_epoch_time_s"]
    steady_epochs_observed = epoch_timing_summary["steady_state_epochs_observed"]
    sampling_source = path_sampling_history
    if not sampling_source and final_sampling_stats:
        sampling_source = [final_sampling_stats]
    path_sampling_summary = summarize_path_sampling_history(
        sampling_source, path_budget=args.path_budget
    )

    best_loss_value = None if not math.isfinite(best_loss) else float(best_loss)
    performance = {
        "run_timestamp": run_timestamp,
        "classification": classification_results,
        "clustering": clustering_result,
        "epochs_ran": epochs_ran,
        "best_train_loss": best_loss_value,
        "timing": {
            "data_loading_time_s": data_loading_seconds,
            "preprocessing_time_s": preprocessing_seconds,
            "model_setup_time_s": model_setup_seconds,
            "training_time_per_epoch_s": epoch_times,
            "avg_training_time_per_epoch_s": avg_epoch_seconds,
            "std_training_time_per_epoch_s": std_epoch_seconds,
            "steady_state_avg_epoch_time_s": steady_avg_epoch_seconds,
            "steady_state_std_epoch_time_s": steady_std_epoch_seconds,
            "steady_state_epochs_observed": steady_epochs_observed,
            "inference_time_s": inference_seconds,
            "evaluation_time_s": evaluation_seconds,
            "training_plus_evaluation_time_s": total_seconds,
            "end_to_end_time_s": end_to_end_seconds,
            "profile_stage_summary": _summarize_stage_profiles(profile_history),
            "profile_training_enabled": bool(args.profile_training),
        },
        "memory": {
            "peak_gpu_memory_mb": training_peak_gpu_memory_mb,
            "peak_gpu_reserved_mb": training_peak_gpu_reserved_mb,
            "peak_inference_gpu_memory_mb": peak_inference_gpu_memory_mb,
        },
        "path_sampling_summary": path_sampling_summary,
        "final_inference_path_sampling_stats": final_sampling_stats,
        "parameter_count": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
        },
        "environment": _environment_metadata(device),
    }

    if args.save_experiment_stats:
        overall_sampling = path_sampling_summary["overall"]
        steady_text = (
            f"steady-state={steady_avg_epoch_seconds:.4f}±{steady_std_epoch_seconds:.4f}s "
            f"(n={steady_epochs_observed})"
            if steady_avg_epoch_seconds is not None
            else "steady-state=N/A (requires >=2 epochs)"
        )

    if return_performance:
        return performance
    return None


if __name__ == '__main__':
    train()
