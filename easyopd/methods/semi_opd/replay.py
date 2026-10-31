# Copyright 2026 EasyOPD Contributors
"""Build a rollout ``DataProto`` from stored offline responses (Semi-OPD replay).

Mirrors the output of ``vLLMRollout.generate_sequences`` (vllm_rollout_spmd.py):
prompts (left-padded, from the dataset), responses right-padded with pad_token_id
to ``response_length``, input_ids = cat, position_ids continued past the prompt,
attention_mask = prompt mask + ``get_response_mask(eos)``. A stored response longer
than ``response_length`` is cut to its prefix -- what an online rollout capped at
that length would have produced -- and then has no EOS, like any truncated rollout.
"""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length

from . import OFFLINE_RESPONSE_KEY


def build_offline_gen_output(gen_batch: DataProto, response_length: int, pad_token_id: int) -> DataProto:
    ntb = dict(gen_batch.non_tensor_batch)
    stored = ntb.pop(OFFLINE_RESPONSE_KEY)
    ntb.pop("raw_prompt_ids", None)  # vllm_rollout pops it too
    idx = gen_batch.batch["input_ids"]
    attention_mask = gen_batch.batch["attention_mask"]
    position_ids = gen_batch.batch["position_ids"]
    bs = idx.size(0)
    if len(stored) != bs:
        raise ValueError(f"[EasyOPD:semi_opd] {len(stored)} stored responses for {bs} prompts (rollout.n must be 1)")
    resp = []
    for r in stored:
        ids = np.asarray(r, dtype=np.int64).tolist()
        if not ids:
            raise ValueError("[EasyOPD:semi_opd] empty stored response")
        resp.append(ids[:response_length])
    response = pad_2d_list_to_length(resp, pad_token_id, max_length=response_length).to(idx.device)
    seq = torch.cat([idx, response], dim=-1)
    delta = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(bs, -1)
    position_ids = torch.cat([position_ids, position_ids[..., -1:] + delta], dim=-1)
    # Mask by the stored length (the sampled ids end at vLLM's stop token, so this equals
    # get_response_mask(eos) without depending on which stop id the student emitted).
    lens = torch.tensor([len(r) for r in resp], device=response.device)
    resp_mask = (torch.arange(response_length, device=response.device)[None, :] < lens[:, None]).to(attention_mask.dtype)
    attention_mask = torch.cat((attention_mask, resp_mask), dim=-1)
    out = DataProto.from_dict(
        tensors={"prompts": idx, "responses": response, "input_ids": seq,
                 "attention_mask": attention_mask, "position_ids": position_ids},
        non_tensors=ntb,
        meta_info={"timing": {}},
    )
    return out
