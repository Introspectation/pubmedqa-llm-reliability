"""Label-position scoring v1: one forward per batch, no generation or parsing."""

import math
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


LABEL_ORDER = ("yes", "no", "maybe")
METRIC_PROTOCOL = "label_score_v1"
JSON_LABEL_PREFIX = '{"label": "'


def choose_labels(log_probabilities):
    values = np.asarray(log_probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite log probabilities with shape (n, 3).")
    return [LABEL_ORDER[index] for index in values.argmax(axis=1)]


def candidate_token_ids(tokenizer, prompt):
    prefix = tokenizer(prompt + JSON_LABEL_PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for label in LABEL_ORDER:
        candidate = tokenizer(prompt + JSON_LABEL_PREFIX + label, add_special_tokens=False)["input_ids"]
        if candidate[:len(prefix)] != prefix or len(candidate) != len(prefix) + 1:
            raise ValueError("Labels must each be one continuation token at this exact JSON position.")
        ids.append(candidate[-1])
    if len(set(ids)) != 3:
        raise ValueError("Candidate labels have duplicate token IDs.")
    return ids


def left_pad_sequences(sequences, pad_token_id):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Each evaluation prompt must contain tokens.")
    width = math.ceil(max(map(len, sequences)) / 8) * 8
    inputs = np.full((len(sequences), width), pad_token_id, dtype=np.int64)
    mask = np.zeros_like(inputs)
    for index, sequence in enumerate(sequences):
        inputs[index, -len(sequence):] = sequence
        mask[index, -len(sequence):] = 1
    positions = np.maximum(mask.cumsum(-1) - 1, 0)
    positions[mask == 0] = 0
    return inputs, mask, positions


def predict_label_scores(model, tokenizer, rows, renderer, batch_size=4):
    """No gold field is accessed here. Return one of three labels per example.

    Values include full-vocabulary log probabilities and probabilities normalized
    over the three allowed label tokens. The latter are conditional readout scores,
    not calibrated clinical confidence. Evaluation never truncates abstracts.
    """
    import torch

    if not rows or batch_size <= 0:
        raise ValueError("Nonempty rows and a positive batch size are required.")
    ids = candidate_token_ids(tokenizer, renderer(rows[0]))
    token_ids = torch.tensor(ids, device="cuda:0")
    outputs = []
    was_training = model.training
    model.eval()
    forward_passes = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            sequences = [tokenizer(renderer(row) + JSON_LABEL_PREFIX, add_special_tokens=False)["input_ids"]
                         for row in batch]
            if any(len(sequence) + 1 > 8192 for sequence in sequences):
                raise ValueError("Full evaluation context exceeds the frozen 8192-token budget.")
            # Left padding makes the final tensor position the label predecessor for every row.
            padded = left_pad_sequences(sequences, tokenizer.pad_token_id)
            inputs, mask, positions = [torch.as_tensor(array, device="cuda:0") for array in padded]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                               logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
                label_log_probs = torch.log_softmax(logits, dim=-1).index_select(-1, token_ids)
                conditional_probs = torch.softmax(label_log_probs, dim=-1)
            forward_passes += 1
            scores = label_log_probs.cpu().numpy()
            probabilities = conditional_probs.cpu().tolist()
            predictions = choose_labels(scores)
            for row, prediction, log_probs, probs in zip(batch, predictions, scores.tolist(), probabilities):
                outputs.append({"pmid": row["pmid"], "prediction": prediction,
                                "label_log_probabilities": dict(zip(LABEL_ORDER, log_probs)),
                                "conditional_label_probabilities": dict(zip(LABEL_ORDER, probs))})
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.train(was_training)
    return outputs, {"runtime_seconds": elapsed, "forward_passes": forward_passes,
                     "batch_size": batch_size, "candidate_token_ids": dict(zip(LABEL_ORDER, ids)),
                     "metric_protocol": METRIC_PROTOCOL, "n": len(rows)}


def score_label_predictions(rows, predictions):
    expected = {row["pmid"] for row in rows}
    by_id = {row["pmid"]: row for row in predictions}
    if not expected or len(expected) != len(rows) or len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Prediction IDs must exactly match the unique scoring IDs.")
    truth = [row["label"] for row in rows]
    predicted = [by_id[row["pmid"]]["prediction"] for row in rows]
    if any(label not in LABEL_ORDER for label in truth + predicted):
        raise ValueError("Unexpected label in three-class scoring.")
    return {"metric_protocol": METRIC_PROTOCOL, "n": len(rows),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=LABEL_ORDER, average="macro", zero_division=0)),
            "confusion_matrix": {"row_labels": list(LABEL_ORDER), "column_labels": list(LABEL_ORDER),
                                 "values": confusion_matrix(truth, predicted, labels=LABEL_ORDER).tolist()},
            "invalid_label_count": 0, "json_parse_failure_count": None,
            "json_parse_failure_applicable": False, "evidence_generation_evaluated": False}
