"""Command-line voice cloning.

Example:
  python clone_cli.py --ref my_voice.wav --text "Hello there" --out hello.wav
  python clone_cli.py --ref my_voice.wav --text-file script.txt --lang hi --out hindi.wav
"""

import argparse
import time

from cloner import LANGUAGES, clone_voice, save_wav


def main():
    p = argparse.ArgumentParser(description="Clone a voice with Chatterbox (free, open source).")
    p.add_argument("--ref", required=True, help="reference voice sample (wav/mp3, 5-20 s)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="text to speak")
    g.add_argument("--text-file", help="file containing text to speak")
    p.add_argument("--out", default="output.wav")
    p.add_argument("--lang", default="en", choices=sorted(LANGUAGES.values()))
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int)
    p.add_argument("--i-have-consent", action="store_true", required=True,
                   help="confirm you own the voice or have the speaker's permission")
    a = p.parse_args()

    text = a.text if a.text else open(a.text_file, encoding="utf-8").read()
    t0 = time.time()
    sr, wav = clone_voice(text, a.ref, language=a.lang, exaggeration=a.exaggeration,
                          cfg_weight=a.cfg_weight, temperature=a.temperature, seed=a.seed)
    save_wav(a.out, sr, wav)
    print(f"Saved {a.out} ({wav.shape[-1] / sr:.1f}s audio) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
