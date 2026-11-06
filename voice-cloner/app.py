"""Free voice cloning web app (Gradio + Chatterbox). NexinEV is the default voice.

Run:  python app.py           -> open http://127.0.0.1:7860
      python app.py --share   -> also get a temporary public link
"""

import argparse
import tempfile

import gradio as gr

from cloner import DEFAULT_VOICE, LANGUAGES, VOICES, clone_voice, pick_device, save_wav

CUSTOM = "Custom (upload / record)"
CONSENT_TEXT = (
    "I own this voice or have the speaker's explicit permission to clone it, "
    "and I will not use it for fraud, impersonation or deception."
)


def run(voice, ref_audio, text, language, exaggeration, cfg_weight, temperature, seed, consent):
    if not text or not text.strip():
        raise gr.Error("Type the text you want spoken.")
    if voice == CUSTOM:
        if not ref_audio:
            raise gr.Error("Upload or record a reference voice sample (5-20 seconds).")
        if not consent:
            raise gr.Error("Please confirm you have permission to clone this voice.")
    sr, wav = clone_voice(
        text,
        ref_audio if voice == CUSTOM else None,
        voice=None if voice == CUSTOM else voice,
        language=LANGUAGES[language],
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        seed=int(seed) if seed else None,
    )
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    return save_wav(out, sr, wav)


def toggle_custom(voice):
    show = voice == CUSTOM
    return gr.update(visible=show), gr.update(visible=show)


with gr.Blocks(title="NexinEV Voice Studio") as demo:
    gr.Markdown(
        f"# 🎙️ NexinEV Voice Studio\n"
        f"Generate voiceovers in the **NexinEV** brand voice, or clone another voice, using "
        f"open-source [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT). "
        f"Running on **{pick_device().upper()}**. Outputs include Resemble's inaudible Perth watermark."
    )
    with gr.Row():
        with gr.Column():
            voice = gr.Dropdown(list(VOICES) + [CUSTOM], value=DEFAULT_VOICE, label="Voice")
            ref = gr.Audio(label="Reference voice (5-20 s, clean, one speaker)",
                           sources=["upload", "microphone"], type="filepath", visible=False)
            text = gr.Textbox(label="Text to speak", lines=5,
                              value="Welcome to NexinEV. Charge smarter, drive further, "
                                    "and power your journey with clean energy.")
            language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
            with gr.Accordion("Advanced", open=False):
                exaggeration = gr.Slider(0.25, 2.0, value=0.5, step=0.05,
                                         label="Exaggeration (emotion intensity)")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                       label="CFG / pace (lower = slower, more deliberate)")
                temperature = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            consent = gr.Checkbox(label=CONSENT_TEXT, visible=False)
            btn = gr.Button("Generate voiceover", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Generated speech", type="filepath")
            gr.Markdown(
                "**Tips:** for custom voices use a clear recording with no music. "
                "On CPU expect roughly 5x real time; a GPU (e.g. free Google Colab) is much faster."
            )
    voice.change(toggle_custom, voice, [ref, consent])
    btn.click(run, [voice, ref, text, language, exaggeration, cfg_weight, temperature, seed, consent], out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    demo.queue().launch(server_name=a.host, server_port=a.port, share=a.share)
