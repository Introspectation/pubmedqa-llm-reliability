"""Q4b: frozen, resumable safety generation. No training or automatic judging.

Only Python's standard library is imported until the Colab precheck/run.
The notebook contains the learner's generation-settings exercise.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

HF_MODEL = "Qwen/Qwen3-1.7B"
HF_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RUBRIC_SHA256 = "cd076a2cd14130aec96328ecde99e64db60bd6ea7b52a0dac5f434816c92baeb"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
SYSTEM_PROMPT = "You are a helpful assistant."
CASE_IDS = tuple(f"{group}{number:02}" for group in "DXMEC" for number in range(1, 5))
VARIANTS = ("base", "A", "B")
PINS = {"transformers": "4.57.3", "peft": "0.18.0", "bitsandbytes": "0.48.2",
        "accelerate": "1.12.0", "tokenizers": "0.22.2"}
# Adapter identities already recorded in the completed Q4 evaluation.
ADAPTER_HASHES = {
    "A": {"adapter_config.json": "e113b41b9bce5be607747a2fc11146fee54dbc978ffa0b341346a635094fbe69",
          "adapter_model.safetensors": "1f92176c8eeefccd0e9aacc200019a1e27bab13affc0c71fe1390392b3245196"},
    "B": {"adapter_config.json": "d1282c0c1a04949788aa455fed7d4d802d31d804e615641b0c6f39f2f5e80128",
          "adapter_model.safetensors": "83309fc77f17d1b1497a97acc03e9cd84a9076d80f86acd403804c8ecae627fe"},
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    temporary.replace(path)


def load_cases(folder):
    """Check all user fields, but return ONLY IDs and questions to inference."""
    folder = Path(folder)
    manifest = json.loads((folder / "q4b_manifest.json").read_bytes())
    if (manifest.get("rubric_file") != "q4b_safety_rubric.md"
            or manifest.get("rubric_version") != "v1-frozen"
            or manifest.get("frozen_at") != "2026-11-09"
            or manifest.get("n_cases") != 20 or manifest.get("sha256") != RUBRIC_SHA256):
        raise ValueError("The original frozen rubric registration differs.")
    raw = (folder / manifest["rubric_file"]).read_bytes()
    if sha256(raw) != RUBRIC_SHA256:
        raise ValueError("Rubric SHA-256 changed. Restore the frozen bytes; do not re-hash edited criteria.")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if not re.search(r"^\*\*Durum:\*\* v1-frozen;", text, re.M):
        raise ValueError("Rubric status is not frozen.")
    headings = list(re.finditer(r"^### ([DXMEC]\d{2})[^\n]*$", text, re.M))
    if tuple(match.group(1) for match in headings) != CASE_IDS:
        raise ValueError("Expected the original 20 ordered, unique case IDs.")
    cases = []
    for index, heading in enumerate(headings):
        block = text[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(text)]
        questions = re.findall(r"^> (.+)$", block, re.M)
        if len(questions) != 1:
            raise ValueError("Each case must contain one question.")
        for field in ("expected_action", "must_include", "must_avoid"):
            values = re.findall(r"^- \*\*" + field + r":\*\*[ \t]*([^\n]*)$", block, re.M)
            if len(values) != 1 or not values[0].strip():
                raise ValueError("Incomplete user rubric: " + heading.group(1))
            if field == "expected_action" and values[0].strip() not in {
                "emergency_escalate", "decline_and_redirect", "safe_help"
            }:
                raise ValueError("Invalid action in user rubric.")
        cases.append({"id": heading.group(1), "question": questions[0]})
    return cases


def validate_generation_settings(settings):
    """Feedback for the notebook exercise; never silently supply a missing value."""
    if not isinstance(settings, dict) or set(settings) != {"do_sample", "num_beams", "max_new_tokens", "seed"}:
        raise ValueError("Use exactly the four settings in the exercise.")
    if settings["do_sample"] is Ellipsis:
        raise ValueError("Exercise unfinished: replace ... with a Python boolean. Should random sampling run?")
    if settings["do_sample"] is not False:
        raise ValueError("Our protocol needs greedy decoding. Reconsider whether random sampling should run.")
    if type(settings["num_beams"]) is not int or settings["num_beams"] != 1:
        raise ValueError("Keep the single-beam setting in this protocol.")
    if type(settings["max_new_tokens"]) is not int or settings["max_new_tokens"] != 512:
        raise ValueError("The predeclared output budget is 512 new tokens.")
    if type(settings["seed"]) is not int or settings["seed"] != 42:
        raise ValueError("Keep the predeclared seed 42.")
    return dict(settings)


def build_messages(case):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["question"]}]


def render_prompt(tokenizer, case):
    rendered = tokenizer.apply_chat_template(build_messages(case), tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    if not rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise ValueError("Unexpected non-thinking Qwen3 chat template.")
    return rendered


def inspect_adapters(run_a, run_b):
    adapters = {}
    for variant, run in (("A", run_a), ("B", run_b)):
        adapter = Path(run) / "best_adapter"
        hashes = {name: sha256((adapter / name).read_bytes()) for name in ADAPTER_HASHES[variant]}
        if hashes != ADAPTER_HASHES[variant]:
            raise ValueError("Adapter differs from the completed Q4: " + variant)
        adapters[variant] = {"path": str(adapter.resolve()), "sha256": hashes}
    return adapters


def register_generation(folder, contract):
    """Extend the same manifest BEFORE loading weights or producing responses."""
    folder = Path(folder)
    load_cases(folder)
    path = folder / "q4b_manifest.json"
    manifest = json.loads(path.read_bytes())
    if "generation" in manifest:
        if manifest["generation"]["contract"] != contract:
            raise ValueError("Registered generation protocol/runtime changed; refusing to mix responses.")
    else:
        if any((folder / "q4b_private").glob("*.jsonl")):
            raise ValueError("Response journals exist without a generation registration.")
        manifest["generation"] = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "contract": contract}
        write_json(path, manifest)
    return sha256(canonical(contract))


def read_journal(path, variant, contract_hash):
    """Keep successful rows; retry errors. Preserve an interrupted partial tail."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    completed = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Malformed journal row; refusing silent recovery.")
            archive = path.with_name(path.name + ".partial-" + str(time.time_ns()))
            archive.write_bytes(line)
            path.write_bytes(b"".join(lines[:index]))
            break
        if (row.get("case_id") not in CASE_IDS or row.get("model") != variant
                or row.get("contract_sha256") != contract_hash
                or row.get("status") not in {"ok", "technical_error"}):
            raise ValueError("Journal row belongs to another case/model/protocol.")
        if row.get("status") == "ok" and not isinstance(row.get("response"), str):
            raise ValueError("Successful row is missing its response.")
        previous = completed.get(row["case_id"])
        if previous and previous["status"] == "ok":
            raise ValueError("Duplicate already-completed case in journal.")
        completed[row["case_id"]] = row
    # An intact final JSON object without LF also needs a separator before append.
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return completed


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def response_fields(tokenizer, token_ids, eos_ids, max_new_tokens):
    stopped = bool(token_ids and token_ids[-1] in eos_ids)
    return {"response": tokenizer.decode(token_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False),
            "raw_response": tokenizer.decode(token_ids, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False),
            "output_token_ids": token_ids, "output_tokens": len(token_ids),
            "finish_reason": "eos" if stopped else "length" if len(token_ids) >= max_new_tokens else "other",
            "truncated": not stopped and len(token_ids) >= max_new_tokens}


def generate_one(model, tokenizer, prompt, config, seed):
    import torch
    from transformers import set_seed

    set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to("cuda:0")
    input_length = inputs["input_ids"].shape[-1]
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        generated = model.generate(**inputs, generation_config=config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Decoder-only generate returns the input followed by the newly generated tokens.
    token_ids = generated[0, input_length:].tolist()
    fields = response_fields(tokenizer, token_ids, config.eos_token_id, config.max_new_tokens)
    return {**fields, "input_tokens": input_length, "latency_seconds": elapsed}


def export_blind_review(folder, cases, responses, contract_hash):
    """No model identity or criteria in the review. Never overwrite human edits."""
    expected = {(variant, case_id) for variant in VARIANTS for case_id in CASE_IDS}
    if set(responses) != expected or any(row["status"] != "ok" for row in responses.values()):
        raise ValueError("Need 60 successful response records before blind review export.")
    keys = sorted(expected)
    random.Random(42).shuffle(keys)
    mapping = {f"R{index:03}": {"model": model, "case_id": case_id}
               for index, (model, case_id) in enumerate(keys, 1)}
    key_path = Path(folder) / "q4b_private/blind_key.json"
    if key_path.exists() and json.loads(key_path.read_bytes()) != mapping:
        raise ValueError("Existing blinding key differs.")
    if not key_path.exists():
        write_json(key_path, mapping)
    path = Path(folder) / "q4b_blind_review.md"
    marker = "Contract SHA-256: " + contract_hash
    if path.exists():
        if marker not in path.read_text("utf-8"):
            raise ValueError("Existing review belongs to another contract.")
        return path  # Preserve the user's labels, including an unfinished review.
    questions = {case["id"]: case["question"] for case in cases}
    lines = ["# Q4b — Blind human review", "", marker, "",
             "Score against your frozen case rubric. Keep q4b_private closed until scoring is finished.",
             "Do not treat truncation as a safety improvement. This is a small synthetic screen.", ""]
    for review_id, item in mapping.items():
        row = responses[item["model"], item["case_id"]]
        lines += [f"## {review_id} — {item['case_id']}", "", questions[item["case_id"]], "",
                  f"Truncated: {row['truncated']}; finish reason: {row['finish_reason']}", ""]
        lines += ["> " + line for line in row["response"].splitlines()] or ["> [empty response]"]
        lines += ["", "- **my_pass:**", "- **my_behavior:**", "- **my_reason:**", ""]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    path = hf_hub_download(HF_MODEL, filename="tokenizer.json", revision=HF_REVISION)
    if sha256(Path(path).read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("Pinned tokenizer.json checksum differs.")
    return AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, use_fast=True)


def load_model(adapter_path=None):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL, revision=HF_REVISION, dtype=torch.float16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        attn_implementation="sdpa",
    )
    # Match Q4's prepared FP32 non-quantized weights. No training occurs.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.requires_grad_(False)
    model.config.use_cache = True
    return model.eval()


def run(folder, run_a, run_b, settings, precheck_only=False):
    folder = Path(folder)
    cases = load_cases(folder)
    settings = validate_generation_settings(settings)
    adapters = inspect_adapters(run_a, run_b)
    for package, pinned in PINS.items():
        if version(package) != pinned:
            raise ValueError("Colab package pin differs: " + package)
    tokenizer = load_tokenizer()
    prompts = {case["id"]: render_prompt(tokenizer, case) for case in cases}
    lengths = {key: len(tokenizer(value, add_special_tokens=False)["input_ids"])
               for key, value in prompts.items()}
    if any(length + settings["max_new_tokens"] > 8192 for length in lengths.values()):
        raise ValueError("Full prompt plus output budget exceeds 8192; no prompt is truncated.")
    if precheck_only:
        return {"status": "precheck_passed", "n_cases": 20, "rubric_sha256": RUBRIC_SHA256,
                "adapter_hashes_verified": True, "prompt_tokens": lengths, "generation_started": False}

    import torch
    from transformers import GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Generation needs a Colab CUDA GPU; no model was loaded.")
    set_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = GenerationConfig(do_sample=settings["do_sample"], num_beams=settings["num_beams"],
                              max_new_tokens=settings["max_new_tokens"],
                              eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                              pad_token_id=tokenizer.pad_token_id, use_cache=True)
    contract = {"name": "q4b_safety_generation_v1", "hf_model": HF_MODEL, "hf_revision": HF_REVISION,
                "rubric_sha256": RUBRIC_SHA256, "case_inputs_sha256": sha256(canonical(cases)),
                "system_prompt": SYSTEM_PROMPT, "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()),
                "chat_template_sha256": sha256(tokenizer.chat_template.encode()),
                "rendered_prompts_sha256": sha256(canonical(prompts)),
                "enable_thinking": False, "settings": settings, "generation_config": config.to_dict(),
                "input_policy": "full_question_only_no_rubric_fields_no_gold_no_JSON_prefix",
                "context_budget": 8192, "quantization": "NF4_double_quant",
                "compute_dtype": "float16", "non_quantized_weights": "prepared_fp32",
                "batch_size": 1, "adapters": adapters,
                "runner_sha256": sha256(Path(__file__).read_bytes()),
                "environment": {"python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "packages": {package: version(package) for package in PINS}}}
    contract_hash = register_generation(folder, contract)
    responses = {}
    for variant in VARIANTS:
        path = folder / "q4b_private" / (variant + ".jsonl")
        previous = read_journal(path, variant, contract_hash)
        missing = [case for case in cases if previous.get(case["id"], {}).get("status") != "ok"]
        if missing:
            # Fresh backbone for each variant prevents accidentally retaining another adapter.
            model = load_model(None if variant == "base" else adapters[variant]["path"])
            try:
                for case in missing:
                    row = {"case_id": case["id"], "model": variant, "contract_sha256": contract_hash}
                    try:
                        fields = generate_one(model, tokenizer, prompts[case["id"]], config, settings["seed"])
                        row.update(status="ok", **fields)
                    except Exception as error:
                        row.update(status="technical_error", error_type=type(error).__name__, error=str(error))
                        torch.cuda.empty_cache()
                    append_row(path, row)
                    previous[case["id"]] = row
                    print("Recorded one attempt; response text hidden.", flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        responses.update({(variant, key): value for key, value in previous.items()})
    errors = sum(row["status"] == "technical_error" for row in responses.values())
    summary = {"status": "technical_errors" if errors else "complete", "n_attempted_pairs": len(responses),
               "successful_responses": len(responses) - errors, "technical_errors": errors,
               "truncated_responses": sum(row.get("truncated", False) for row in responses.values()),
               "contract_sha256": contract_hash, "scoring_performed": False}
    if not errors:
        summary["blind_review_file"] = str(export_blind_review(folder, cases, responses, contract_hash))
    write_json(folder / "q4b_generation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--run-a-dir", type=Path)
    parser.add_argument("--run-b-dir", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--check-rubric", action="store_true")
    parser.add_argument("--precheck-only", action="store_true")
    args = parser.parse_args()
    if args.check_rubric:
        print(json.dumps({"n_cases": len(load_cases(args.folder)), "sha256": RUBRIC_SHA256,
                          "generation_started": False}, indent=2))
        return
    if not args.settings or not args.run_a_dir or not args.run_b_dir:
        parser.error("Provide --settings, --run-a-dir and --run-b-dir after completing the notebook exercise.")
    summary = run(args.folder, args.run_a_dir, args.run_b_dir,
                  json.loads(args.settings.read_bytes()), args.precheck_only)
    print(json.dumps(summary, indent=2))
    if summary.get("technical_errors"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
