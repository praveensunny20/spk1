# 🎙️ Free Voice Cloner

Clone a voice from a 5–20 second sample and make it say anything. The tool is
100% free and open source, and it runs entirely on your own machine or on a
free Google Colab GPU. You don't need an API key, a Hugging Face token or an
account.

It's built on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox),
which is MIT licensed, so commercial use is allowed. It supports **23 languages**,
including English and Hindi.

## Features
- Zero-shot cloning: upload or record a short sample, with no training needed
- Web UI (Gradio) and a command-line script
- 23 languages, plus controls for emotion, pace and temperature
- Splits long text into chunks automatically
- Asks for consent before cloning, and every output carries Resemble's
  inaudible Perth watermark

## Option 1: Run on your PC

Requires Python 3.10 or 3.11. A GPU is optional but much faster.

```bash
cd voice-cloner
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # open http://127.0.0.1:7860
```

The first run downloads about 3 GB of model weights from Hugging Face
(public, no token needed). After that the weights are cached.

Command line:

```bash
python clone_cli.py --ref my_voice.wav --text "Hello, this is my cloned voice." --out hello.wav --i-have-consent
python clone_cli.py --ref my_voice.wav --text "नमस्ते, यह मेरी आवाज़ है।" --lang hi --out hindi.wav --i-have-consent
```

## Option 2: Free GPU on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com), go to
   **Upload**, and choose `voice_cloner_colab.ipynb` from this folder.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all the cells, then open the public `gradio.live` link it prints.

## Getting good results
- Use **clean audio**: one speaker, no music, no echo, 10–20 seconds long.
- If the voice sounds too fast or rushed, lower **CFG** to about 0.3.
- For more dramatic or emotional speech, raise **Exaggeration** to 0.7 or
  higher and lower CFG to about 0.3.
- To get the same output again, set a **Seed**.

## Speed
| Hardware | Rough time for 10 s of speech |
|---|---|
| NVIDIA GPU / Colab T4 | 3–8 s |
| Modern 4–8 core CPU | 30–90 s |

## ⚠️ Responsible use
Only clone your own voice or a voice you have **explicit permission** to use.
Using cloned voices for fraud, impersonation, scams or deception is illegal in
many countries.
