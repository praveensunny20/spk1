"""Free voice cloning web app (Gradio + Chatterbox).

Run:  python app.py           -> open http://127.0.0.1:7860
      python app.py --share   -> also get a temporary public link
"""

import argparse
import tempfile

import gradio as gr

from cloner import LANGUAGES, clone_voice, pick_device, save_wav

CONSENT_TEXT = (
    "I own this voice or have the speaker's explicit permission to clone it, "
    "and I will not use it for fraud, impersonation or deception."
)


def run(ref_audio, text, language, exaggeration, cfg_weight, temperature, seed, consent):
    if not consent:
        raise gr.Error("Please confirm you have permission to clone this voice.")
    if not ref_audio:
        raise gr.Error("Upload or record a reference voice sample (5-20 seconds).")
    if not text or not text.strip():
        raise gr.Error("Type the text you want spoken.")
    sr, wav = clone_voice(
        text,
        ref_audio,
        language=LANGUAGES[language],
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        seed=int(seed) if seed else None,
    )
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    return save_wav(out, sr, wav)


with gr.Blocks(title="Free Voice Cloner") as demo:
    gr.Markdown(
        f"# 🎙️ Free Voice Cloner\n"
        f"Clone a voice from a short sample using open-source "
        f"[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT). "
        f"Running on **{pick_device().upper()}**. Outputs include Resemble's "
        f"inaudible Perth watermark."
    )
    with gr.Row():
        with gr.Column():
            ref = gr.Audio(label="Reference voice (5-20 s, clean, one speaker)",
                           sources=["upload", "microphone"], type="filepath")
            text = gr.Textbox(label="Text to speak", lines=5,
                              value="Hello! This is my cloned voice, made with a free open-source tool.")
            language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
            with gr.Accordion("Advanced", open=False):
                exaggeration = gr.Slider(0.25, 2.0, value=0.5, step=0.05,
                                         label="Exaggeration (emotion intensity)")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                       label="CFG / pace (lower = slower, more deliberate)")
                temperature = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            consent = gr.Checkbox(label=CONSENT_TEXT)
            btn = gr.Button("Clone voice", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Cloned speech", type="filepath")
            gr.Markdown(
                "**Tips:** use a clear recording with no music or background noise. "
                "On CPU expect roughly 1x-3x real time; a GPU (e.g. free Google Colab) is much faster."
            )
    btn.click(run, [ref, text, language, exaggeration, cfg_weight, temperature, seed, consent], out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    demo.queue().launch(server_name=a.host, server_port=a.port, share=a.share)
