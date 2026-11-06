"""Command-line voice cloning.

Examples:
  # Default NexinEV brand voice - no reference needed
  python clone_cli.py --text "Welcome to NexinEV" --out welcome.wav
  python clone_cli.py --text-file script.txt --lang hi --out hindi.wav

  # Any other voice (needs consent)
  python clone_cli.py --ref my_voice.wav --text "Hello there" --out hello.wav --i-have-consent
"""

import argparse
import sys
import time

from cloner import DEFAULT_VOICE, LANGUAGES, VOICES, clone_voice, save_wav


def main():
    p = argparse.ArgumentParser(description="Clone a voice with Chatterbox (free, open source).")
    p.add_argument("--voice", default=DEFAULT_VOICE, choices=list(VOICES),
                   help=f"built-in voice preset (default: {DEFAULT_VOICE})")
    p.add_argument("--ref", help="custom reference voice sample (wav/mp3, 5-20 s); overrides --voice")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="text to speak")
    g.add_argument("--text-file", help="file containing text to speak")
    p.add_argument("--out", default="output.wav")
    p.add_argument("--lang", default="en", choices=sorted(LANGUAGES.values()))
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int)
    p.add_argument("--i-have-consent", action="store_true",
                   help="required with --ref: confirm you own the voice or have the speaker's permission")
    a = p.parse_args()
    if a.ref and not a.i_have_consent:
        sys.exit("Using a custom --ref voice requires --i-have-consent.")

    text = a.text if a.text else open(a.text_file, encoding="utf-8").read()
    t0 = time.time()
    sr, wav = clone_voice(text, a.ref, language=a.lang, exaggeration=a.exaggeration,
                          cfg_weight=a.cfg_weight, temperature=a.temperature, seed=a.seed,
                          voice=a.voice)
    save_wav(a.out, sr, wav)
    print(f"Saved {a.out} ({wav.shape[-1] / sr:.1f}s audio) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
