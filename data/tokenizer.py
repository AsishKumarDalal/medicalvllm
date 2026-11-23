"""GPT-2 BPE tokenization matching minillm's SFT format.

minillm's SFT prompt is (see sft/sft_data.py)::

    Instruction:
    <instruction>

    Context:
    <context>

    Response:

with the response + EOT as the only supervised tokens. We reuse the exact same
template for medical VQA so the SFT checkpoint's learned instruction-following
behavior transfers directly.
"""

from functools import lru_cache

import tiktoken

EOT_TOKEN_ID = 50256


@lru_cache(maxsize=8)
def get_encoder(name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def format_vqa_prompt(question: str, context: str | None = None) -> str:
    """Build the instruction prompt (without the answer)."""
    instruction = " ".join(
        "Answer the following medical question about the image with a short, direct answer.".split()
    )
    question = " ".join(question.split())
    if context:
        context = " ".join(context.split())
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Response:\n"
        )
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Question:\n{question}\n\n"
        "Response:\n"
    )


def format_caption_prompt() -> str:
    """Stage 1: describe-the-image prompt in the same instruction style."""
    return (
        "Instruction:\nDescribe the medical image in one sentence.\n\n"
        "Response:\n"
    )


def encode_example(
    prompt: str,
    response: str,
    budget: int,
    encoder: tiktoken.Encoding | None = None,
    eot: int = EOT_TOKEN_ID,
):
    """Encode ``prompt + response + EOT`` with prompt tokens masked.

    Triage (mirrors minillm's SFTData):
      1. full prompt + response fits -> keep;
      2. otherwise try dropping the ``Context:`` block;
      3. otherwise encode only question + response (drop instruction too);
      4. still too long -> return None.

    Returns ``(ids, labels, triage)``.
    """
    enc = encoder or get_encoder()
    prompt = " ".join(prompt.split()) if prompt else prompt
    response = " ".join(response.split())
    if not response or not prompt:
        return None

    p_ids = enc.encode(prompt)
    r_ids = enc.encode(response) + [eot]

    def pack(p):
        return p + r_ids, [-100] * len(p) + r_ids

    if len(p_ids) + len(r_ids) <= budget:
        ids, labels = pack(p_ids)
        return ids, labels, "full"

    # Drop the Context: block if present.
    short = _drop_context_block(prompt)
    if short != prompt:
        s_ids = enc.encode(short)
        if len(s_ids) + len(r_ids) <= budget:
            ids, labels = pack(s_ids)
            return ids, labels, "no_context"

    # Last resort: question + response only.
    q = _extract_after(prompt, "Question:")
    if q:
        q_ids = enc.encode("Question:\n" + q + "\n\nResponse:\n")
        if len(q_ids) + len(r_ids) <= budget:
            ids, labels = pack(q_ids)
            return ids, labels, "question_only"
    return None


def _drop_context_block(prompt: str) -> str:
    start = prompt.find("Context:\n")
    if start < 0:
        return prompt
    end = prompt.find("\n\n", start)
    if end < 0:
        return prompt
    return prompt[:start] + prompt[end + 2 :]


def _extract_after(prompt: str, marker: str) -> str | None:
    start = prompt.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = prompt.find("\n\n", start)
    return prompt[start:end].strip() if end > 0 else prompt[start:].strip()
