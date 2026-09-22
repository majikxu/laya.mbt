"""Dump golden fixtures from laya-mlx so laya.mbt can be checked for numerical parity.

Run:  uv run python dump_fixtures.py <model-dir> <out.json> [--full]

Everything is computed in float32 on the CPU, which is the configuration
`validation.json` in the upstream checkpoints reports agreeing with PyTorch to
~5e-6 on probabilities. `--full` additionally records per-stage activations for
a single short case, which is what you want while bringing the port up; without
it only tokens plus final logits and the public result are recorded.
"""

import argparse
import json
import sys

import mlx.core as mx
import numpy as np

import laya_mlx as laya
from laya_mlx.agent import collate_items
from laya_mlx.common import QTYPES, serialize_state

CASES = {
    "en_support": {
        "state": "I was billed twice this month. Please refund the duplicate charge.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Who should handle this?",
                "criteria": ["billing", "technical", "sales"],
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "urgent", "critical"],
            },
            "refund": {
                "type": "noul",
                "instructions": "The customer is asking for a refund.",
            },
        },
    },
    "zh_support": {
        "state": "我这个月被重复扣款了两次，请把多收的那笔退给我。",
        "questions": {
            "部门": {
                "type": "choice",
                "instructions": "这条工单该由谁处理？",
                "criteria": ["账务", "技术", "销售"],
            },
            "退款": {
                "type": "noul",
                "instructions": "客户正在要求退款。",
            },
        },
    },
    "json_state": {
        "state": {"plan": "pro", "seats": 12, "overdue_days": 41, "region": "EU"},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What should the billing system do next?",
                "criteria": {
                    "dunning": "send a payment reminder",
                    "suspend": "suspend the workspace",
                    "wait": "take no action yet",
                },
            }
        },
    },
    "single_option": {
        "state": "ticket closed by the reporter",
        "questions": {
            "only": {
                "type": "choice",
                "instructions": "Pick the label.",
                "criteria": ["closed"],
            }
        },
    },
    "tiny": {
        "state": "refund me",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "The customer wants money back.",
            }
        },
    },
}


def to_list(x):
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


def stage_activations(agent, batch):
    """Re-run the forward pass stage by stage, recording each intermediate."""
    model = agent.model
    cfg = model.encoder.config
    tensors = {k: mx.array(v) for k, v in batch.items()}
    input_ids = tensors["input_ids"]
    attention_mask = tensors["attention_mask"]
    stages = {}

    from laya_mlx.model import attention_masks

    x = model.encoder.embeddings(input_ids)
    stages["embeddings"] = to_list(x)
    masks = attention_masks(attention_mask, cfg.local_attention)
    for i, layer in enumerate(model.encoder.layers):
        x = layer(x, masks[layer.attention_type])
        if i in (0, 1, len(model.encoder.layers) - 1):
            stages[f"encoder_layer_{i}"] = to_list(x)
    x = model.encoder.final_norm(x)
    stages["encoder_out"] = to_list(x)

    h = x + model.type_emb(tensors["qtype"])[:, None, :]
    stages["with_type_emb"] = to_list(h)
    head_mask = attention_mask[:, None, None, :].astype(mx.bool_)
    for i, layer in enumerate(model.head.layers):
        h = layer(h, head_mask)
        stages[f"head_layer_{i}"] = to_list(h)

    markers = h[mx.arange(h.shape[0])[:, None], mx.maximum(tensors["marker_pos"], 0)]
    stages["marker_rows"] = to_list(markers)
    stages["scorer_logits"] = to_list(model.scorer(markers).squeeze(-1).astype(mx.float32))
    mx.eval(stages)
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--full", action="store_true", help="also record per-stage activations")
    args = ap.parse_args()

    agent = laya.load(args.model_dir, device="cpu", dtype="float32")
    enc = agent.encoder_cfg

    out = {
        "model_dir": args.model_dir,
        "encoder": {
            "hidden_size": enc["hidden_size"],
            "num_hidden_layers": enc["num_hidden_layers"],
            "num_attention_heads": enc["num_attention_heads"],
            "intermediate_size": enc["intermediate_size"],
            "vocab_size": enc["vocab_size"],
        },
        "special_tokens": {
            "cls": agent.tok.cls_token_id,
            "sep": agent.tok.sep_token_id,
            "pad": agent.tok.pad_token_id,
            "mask": agent.tok.mask_token_id,
        },
        "cases": {},
    }

    for name, case in CASES.items():
        items, internal = agent.prepare(case["state"], case["questions"])
        batch = collate_items(items, agent.tok.pad_token_id, max_length=agent.cfg["max_len"])
        logits, act = agent.forward(batch)
        record = {
            "state": case["state"],
            # laya.mbt's API takes the state as text, so record exactly what the
            # Python runtime serialises a structured state to.
            "state_text": serialize_state(case["state"]),
            "questions": case["questions"],
            "result": agent.predict(case["state"], case["questions"]),
            "items": [
                {
                    "qid": qid,
                    "ids": item["ids"],
                    "markers": item["markers"],
                    "qtype": item["qtype"],
                    "options": None,
                }
                for qid, item in zip(case["questions"], items)
            ],
            "logits": np.asarray(logits, dtype=np.float64).tolist(),
            "action_logits": np.asarray(act, dtype=np.float64).tolist(),
        }
        from laya_mlx.common import render_options

        for entry, q in zip(record["items"], internal):
            entry["options"] = render_options(q)
        if args.full and name == "tiny":
            record["stages"] = stage_activations(agent, batch)
            record["stage_shape"] = [int(v) for v in batch["input_ids"].shape]
        out["cases"][name] = record
        print(f"  {name}: {len(items)} question(s), seq_len={max(len(i['ids']) for i in items)}",
              file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
