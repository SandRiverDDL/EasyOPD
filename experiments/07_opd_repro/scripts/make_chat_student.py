# Copyright 2026 EasyOPD Contributors
"""Make a chat-ready view of a Qwen3 *Base* checkpoint: same weights, ``<|im_end|>`` as EOS.

New script. Why: ``Qwen3-*-Base`` ships ``eos_token=<|endoftext|>`` (151643) in
both tokenizer_config.json and generation_config.json, while the chat template
ends every assistant turn with ``<|im_end|>`` (151645). verl builds the response
mask from ``tokenizer.eos_token_id`` and vLLM stops on the generation-config EOS,
so with the stock files a Base student would keep generating after
``<|im_end|>`` and the post-turn junk would be trained on. The output directory
symlinks every weight file and rewrites only the two json files.

Usage::

    python make_chat_student.py /mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base \
        /mnt/NLU_disk17/chenhaoyu02/models/Qwen3-1.7B-Base-imend
"""

import json
import os
import sys

IM_END, ENDOFTEXT = "<|im_end|>", "<|endoftext|>"
IM_END_ID, ENDOFTEXT_ID = 151645, 151643


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name in ("tokenizer_config.json", "generation_config.json") or name.startswith("."):
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, name), link)

    with open(os.path.join(src, "tokenizer_config.json")) as f:
        tc = json.load(f)
    added = {v["content"]: int(k) for k, v in tc.get("added_tokens_decoder", {}).items()}
    assert added.get(IM_END) == IM_END_ID and added.get(ENDOFTEXT) == ENDOFTEXT_ID, "not a Qwen3 tokenizer"
    tc["eos_token"] = IM_END
    tc["pad_token"] = ENDOFTEXT
    with open(os.path.join(dst, "tokenizer_config.json"), "w") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)

    gc_path = os.path.join(src, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.update({"bos_token_id": ENDOFTEXT_ID, "eos_token_id": [IM_END_ID, ENDOFTEXT_ID], "pad_token_id": ENDOFTEXT_ID})
    gc.pop("max_new_tokens", None)
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(dst)
    assert tok.eos_token_id == IM_END_ID and tok.pad_token_id == ENDOFTEXT_ID, (tok.eos_token_id, tok.pad_token_id)
    print(f"[make_chat_student] {dst}: eos={tok.eos_token}({tok.eos_token_id}) pad={tok.pad_token}({tok.pad_token_id})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
