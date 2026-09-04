import hashlib
from pathlib import Path

import torch


PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_ENCODER_ID = "wan22ti2v5b"
DEFAULT_LTX_ENCODER_ID = "ltx-video-t5"


def format_text_prompt(task: str, prompt_template: str | None = None) -> str:
    task = str(task)
    template = PROMPT_TEMPLATE if not prompt_template else str(prompt_template)
    return template.format(task=task, instruction=task)


def text_cache_path(
    cache_dir: str | Path,
    prompt: str,
    context_len: int = DEFAULT_CONTEXT_LEN,
    encoder_id: str = DEFAULT_ENCODER_ID,
) -> Path:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{prompt_hash}.t5_len{int(context_len)}.{encoder_id}.pt"


def load_text_embedding_cache(
    cache_dir: str | Path,
    task: str | None = None,
    context_len: int = DEFAULT_CONTEXT_LEN,
    encoder_id: str = DEFAULT_ENCODER_ID,
    prompt_template: str | None = None,
    *,
    prompt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a cached text embedding.

    Pass ``task`` to format it through ``prompt_template`` first, or pass an
    already-formatted ``prompt`` (e.g. the unconditional ``""``) to read it verbatim.
    """
    if prompt is None:
        prompt = format_text_prompt(task, prompt_template=prompt_template)
    payload = torch.load(text_cache_path(cache_dir, prompt, context_len, encoder_id), map_location="cpu")
    context = payload["context"]
    mask = payload["mask"].bool()
    if context.shape[0] != int(context_len) or mask.shape[0] != int(context_len):
        raise ValueError(f"Text cache length mismatch for prompt: {prompt}")
    return context, mask


def maybe_load_text_embedding_cache(
    cache_dir: str | Path | None,
    task: str | None = None,
    context_len: int = DEFAULT_CONTEXT_LEN,
    encoder_id: str = DEFAULT_ENCODER_ID,
    required: bool = False,
    prompt_template: str | None = None,
    *,
    prompt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Best-effort variant of :func:`load_text_embedding_cache`.

    Returns ``None`` on cache miss unless ``required`` is set. Accepts either a
    ``task`` (formatted via ``prompt_template``) or a verbatim ``prompt``.
    """
    if not cache_dir:
        return None
    try:
        return load_text_embedding_cache(
            cache_dir,
            task,
            context_len=context_len,
            encoder_id=encoder_id,
            prompt_template=prompt_template,
            prompt=prompt,
        )
    except FileNotFoundError:
        if required:
            raise
        return None
