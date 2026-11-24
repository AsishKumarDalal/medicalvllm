"""Tokenization for both the GPT-2 (minillm) and Qwen LLM backends.

The text format is shared::

    Instruction:
    <instruction>

    Context:
    <context>

    Question:
    <question>

    Response:
    <answer><EOT>

Only ``<answer>`` + ``<EOT>`` are supervised. For GPT-2 we keep the exact
tiktoken BPE used by minillm's SFT checkpoint. For Qwen we use the model's own
HuggingFace tokenizer (no special tokens added to the prompt) and the model's
native EOS as the answer terminator.

The active backend is selected once per run, before any data stream is built:

    from data.tokenizer import set_tokenizer_backend
    set_tokenizer_backend("qwen")            # or "gpt2" (the default)

``get_encoder()``, ``get_eot_id()`` and ``get_pad_id()`` then return the right
encoder / ids for the active backend, so the rest of the codebase stays
backend-agnostic.
"""

from functools import lru_cache

import tiktoken

# GPT-2 / minillm constants (kept as aliases for backward compatibility).
GPT2_EOT_TOKEN_ID = 50256
EOT_TOKEN_ID = GPT2_EOT_TOKEN_ID

QWEN_MODEL = "Qwen/Qwen3-0.6B"
# Qwen3 tokenizer: <|endoftext|>=151643 (pad), <|im_end|>=151645 (EOS).
QWEN_PAD_TOKEN_ID = 151643
QWEN_EOT_TOKEN_ID = 151645

_BACKEND = "gpt2"
_QWEN_MODEL_NAME = QWEN_MODEL
_QWEN_LOADED_NAME = None
_QWEN_TOKENIZER = None


class HFTokenizerAdapter:
    """Small ``encode``/``decode`` shim over a HuggingFace tokenizer.

    Exposes the same two methods the rest of the pipeline used from tiktoken's
    ``Encoding``. Never adds BOS/EOS to a prompt; EOT is appended explicitly by
    :func:`encode_example`.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)


# ------------------------------------------------------------- backend state
def set_tokenizer_backend(
    backend: str = "gpt2",
    model_name: str | None = None,
    tokenizer=None,
) -> str:
    """Select the tokenizer backend for this process.

    ``tokenizer`` can be supplied to inject a pre-built HF tokenizer (used by
    offline tests so no weights/tokenizer are downloaded).
    """
    global _BACKEND, _QWEN_MODEL_NAME, _QWEN_LOADED_NAME, _QWEN_TOKENIZER

    backend = str(backend).lower()
    if backend not in {"gpt2", "qwen"}:
        raise ValueError(f"unknown tokenizer backend: {backend!r}")

    if backend == "qwen":
        _QWEN_MODEL_NAME = str(model_name or _QWEN_MODEL_NAME)
        if tokenizer is not None:
            _QWEN_TOKENIZER = tokenizer
            _QWEN_LOADED_NAME = _QWEN_MODEL_NAME
        elif _QWEN_LOADED_NAME != _QWEN_MODEL_NAME or _QWEN_TOKENIZER is None:
            from transformers import AutoTokenizer

            _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(_QWEN_MODEL_NAME)
            _QWEN_LOADED_NAME = _QWEN_MODEL_NAME

    _BACKEND = backend
    _tiktoken_encoder.cache_clear()
    return _BACKEND


def get_backend() -> str:
    return _BACKEND


def get_qwen_model_name() -> str:
    return _QWEN_MODEL_NAME


def _get_qwen_tokenizer():
    if _QWEN_TOKENIZER is None:
        set_tokenizer_backend("qwen", model_name=_QWEN_MODEL_NAME)
    return _QWEN_TOKENIZER


def get_eot_id() -> int:
    """Id of the answer terminator for the active backend."""
    if _BACKEND == "qwen":
        tok = _get_qwen_tokenizer()
        eos = getattr(tok, "eos_token_id", None)
        return int(eos) if eos is not None else QWEN_EOT_TOKEN_ID
    return GPT2_EOT_TOKEN_ID


def get_pad_id() -> int:
    """Id used to right-pad batches for the active backend."""
    if _BACKEND == "qwen":
        tok = _get_qwen_tokenizer()
        pad = getattr(tok, "pad_token_id", None)
        if pad is not None:
            return int(pad)
        return get_eot_id()
    return GPT2_EOT_TOKEN_ID


# ------------------------------------------------------------- encoders
@lru_cache(maxsize=8)
def _tiktoken_encoder(name: str = "gpt2"):
    return tiktoken.get_encoding(name)


def get_encoder(name: str = "gpt2"):
    """Return an encoder with ``encode(text)`` / ``decode(ids)`` methods."""
    if _BACKEND == "qwen":
        return HFTokenizerAdapter(_get_qwen_tokenizer())
    return _tiktoken_encoder(name)


# ------------------------------------------------------------- formatting
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
    encoder=None,
    eot: int | None = None,
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
    if eot is None:
        eot = get_eot_id()
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
