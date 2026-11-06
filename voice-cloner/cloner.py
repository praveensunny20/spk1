"""Shared voice cloning logic built on Resemble AI's open-source Chatterbox (MIT).

Models are downloaded once from Hugging Face (public, no token needed) and
cached in ~/.cache/huggingface. Runs on CPU, CUDA GPU, or Apple Silicon (MPS).
"""

from __future__ import annotations

import re

import torch
import torchaudio as ta

# Languages supported by the multilingual Chatterbox model.
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Arabic": "ar", "Chinese": "zh",
    "Danish": "da", "Dutch": "nl", "Finnish": "fi", "French": "fr",
    "German": "de", "Greek": "el", "Hebrew": "he", "Italian": "it",
    "Japanese": "ja", "Korean": "ko", "Malay": "ms", "Norwegian": "no",
    "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Spanish": "es",
    "Swahili": "sw", "Swedish": "sv", "Turkish": "tr",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_models: dict[str, object] = {}


def get_model(language: str = "en"):
    """English uses the original model; other languages use the multilingual one."""
    key = "en" if language == "en" else "mtl"
    if key not in _models:
        device = pick_device()
        if key == "en":
            from chatterbox.tts import ChatterboxTTS
            _models[key] = ChatterboxTTS.from_pretrained(device=device)
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            # Pass a plain string: the loader checks `device in ["cpu", "mps"]`
            # to decide whether to remap CUDA-saved weights onto the CPU.
            _models[key] = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _models[key]


def split_text(text: str, max_chars: int = 250) -> list[str]:
    """Split long text into sentence chunks so long scripts don't degrade."""
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


@torch.inference_mode()
def clone_voice(
    text: str,
    reference_audio: str,
    language: str = "en",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    seed: int | None = None,
) -> tuple[int, torch.Tensor]:
    """Speak `text` in the voice of `reference_audio`. Returns (sample_rate, wav)."""
    if not text.strip():
        raise ValueError("Text is empty.")
    if seed is not None:
        torch.manual_seed(seed)
    model = get_model(language)
    kwargs = dict(
        audio_prompt_path=reference_audio,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    if language != "en":
        kwargs["language_id"] = language
    pieces = []
    silence = torch.zeros(1, int(model.sr * 0.25))
    for chunk in split_text(text):
        wav = model.generate(chunk, **kwargs).cpu()
        pieces += [wav, silence]
    return model.sr, torch.cat(pieces[:-1], dim=-1)


def save_wav(path: str, sr: int, wav: torch.Tensor) -> str:
    ta.save(path, wav, sr)
    return path
