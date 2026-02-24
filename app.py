import torch
import gradio as gr
from model import SLM, SLMConfig
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import os

# ── 1. Load Model & Config ───────────────────────────────────────────────────
config = SLMConfig(
    vocab_size=16384, n_layer=10, n_head=8,
    n_embd=512, block_size=256, dropout=0.0
)
model = SLM(config)

weights_path = "vital_lm_50m_weights.pt"
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ── 2. Load Tokenizer ────────────────────────────────────────────────────────
tokenizer_raw = ByteLevelBPETokenizer("vocab_50m.json", "merges_50m.txt")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_raw,
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)

# ── 3. Generation Logic ──────────────────────────────────────────────────────
def respond(message, history, system_message, max_tokens, temperature, top_k):
    prompt = f"{system_message}\n\nPatient: {message}\nDoctor:"
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    generated_text = ""
    for _ in range(max_tokens):
        cond_ids = input_ids[:, -256:]
        with torch.no_grad():
            logits, _ = model(cond_ids)
            logits = logits[:, -1, :] / (temperature + 1e-5)

            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        new_text = tokenizer.decode([next_token.item()])
        generated_text += new_text
        yield generated_text

# ── 4. Custom CSS ────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Syne:wght@700;800&display=swap');
:root {
  --bg:        #080d1a;
  --bg2:       #0d1225;
  --surface:   #111827;
  --surface2:  #1a2235;
  --border:    rgba(56,189,248,0.12);
  --border2:   rgba(56,189,248,0.22);
  --accent:    #38bdf8;
  --accent2:   #818cf8;
  --green:     #34d399;
  --red:       #f87171;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --radius:    14px;
  --font-h:    'Syne', sans-serif;
  --font-b:    'DM Sans', sans-serif;
}
/* ── Reset & base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: var(--bg) !important;
  font-family: var(--font-b) !important;
  color: var(--text) !important;
  min-height: 100vh;
}
/* Ambient background glow */
.gradio-container::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 55% 45% at 15% 15%, rgba(56,189,248,0.06) 0%, transparent 65%),
    radial-gradient(ellipse 45% 55% at 85% 85%, rgba(129,140,248,0.06) 0%, transparent 65%),
    radial-gradient(ellipse 35% 35% at 50% 50%, rgba(52,211,153,0.03) 0%, transparent 65%);
}
/* ── HEADER ─────────────────────────────────────────────────── */
#vitallm-header {
  position: relative;
  text-align: center;
  padding: 56px 24px 40px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(56,189,248,0.05) 0%, transparent 100%);
}
.vl-badge {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(52,211,153,0.1);
  border: 1px solid rgba(52,211,153,0.28);
  color: var(--green);
  font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.13em; text-transform: uppercase;
  padding: 5px 16px; border-radius: 999px;
  margin-bottom: 22px;
}
.vl-pulse {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green);
  animation: vlpulse 2.2s ease infinite;
}
@keyframes vlpulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.7); }
}
#vitallm-header h1 {
  font-family: var(--font-h) !important;
  font-size: clamp(2rem, 5vw, 3.4rem) !important;
  font-weight: 800 !important;
  letter-spacing: -0.025em; line-height: 1.05;
  margin: 0 0 14px !important;
  background: linear-gradient(130deg, #f0f9ff 0%, var(--accent) 45%, var(--accent2) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
#vitallm-header p.vl-sub {
  color: var(--muted); font-size: 15px;
  max-width: 520px; margin: 0 auto; line-height: 1.65;
}
/* Stat chips */
.vl-stats {
  display: flex; justify-content: center; gap: 12px;
  flex-wrap: wrap; margin-top: 30px;
}
.vl-stat {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 20px;
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  min-width: 90px;
}
.vl-stat-num {
  font-family: var(--font-h);
  font-size: 1.25rem; font-weight: 700;
  color: var(--accent);
}
.vl-stat-label {
  font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted);
}
/* ── LAYOUT ─────────────────────────────────────────────────── */
.vl-main {
  max-width: 1160px;
  margin: 0 auto;
  padding: 28px 20px 48px;
}
/* ── CHAT PANEL ─────────────────────────────────────────────── */
.vl-chat-col { flex: 3 !important; }
.vl-settings-col { flex: 1.1 !important; min-width: 270px; }
/* Chatbot container */
#vitallm-chatbot {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}
#vitallm-chatbot > div {
  background: transparent !important;
  border: none !important;
}
/* Bubbles */
.message.user {
  background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(56,189,248,0.07)) !important;
  border: 1px solid rgba(56,189,248,0.22) !important;
  border-radius: 14px 14px 4px 14px !important;
  color: var(--text) !important; font-size: 14px !important;
}
.message.bot {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px 14px 14px 4px !important;
  color: var(--text) !important; font-size: 14px !important;
  line-height: 1.65 !important;
}
/* ── INPUT ROW ──────────────────────────────────────────────── */
#vl-input-row {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 14px !important;
  margin-top: 12px !important;
  gap: 10px !important;
}
#vl-input-row textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font-b) !important;
  font-size: 14px !important;
  resize: none !important;
  transition: border-color .2s, box-shadow .2s;
}
#vl-input-row textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
  outline: none !important;
}
#vl-input-row textarea::placeholder { color: var(--muted) !important; }
/* ── BUTTONS ────────────────────────────────────────────────── */
button.primary, #send-btn {
  background: linear-gradient(135deg, var(--accent), #6366f1) !important;
  border: none !important; border-radius: 10px !important;
  color: #fff !important; font-weight: 600 !important;
  font-family: var(--font-b) !important;
  letter-spacing: 0.02em;
  transition: opacity .2s, transform .15s !important;
  box-shadow: 0 4px 14px rgba(56,189,248,0.25) !important;
}
button.primary:hover, #send-btn:hover {
  opacity: .88 !important; transform: translateY(-1px) !important;
}
button.secondary, #clear-btn {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  color: var(--muted) !important; font-family: var(--font-b) !important;
  transition: border-color .2s, color .2s !important;
}
button.secondary:hover, #clear-btn:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}
/* ── EXAMPLES ───────────────────────────────────────────────── */
.vl-examples { margin-top: 14px; }
.vl-examples .examples-holder table td {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--muted) !important; font-size: 13px !important;
  transition: all .2s !important; cursor: pointer;
}
.vl-examples .examples-holder table td:hover {
  background: var(--surface2) !important;
  border-color: var(--accent) !important; color: var(--accent) !important;
}
/* ── SETTINGS PANEL ─────────────────────────────────────────── */
.vl-settings {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 22px !important;
  height: fit-content;
}
.vl-settings label, .vl-settings .svelte-1gfkn6j {
  color: var(--text) !important;
  font-size: 13px !important; font-weight: 500 !important;
}
.vl-settings textarea, .vl-settings input[type="text"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font-b) !important; font-size: 13px !important;
}
.vl-settings textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}
input[type="range"] { accent-color: var(--accent) !important; }
/* Divider inside settings */
.vl-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 16px 0;
}
/* Slider value display */
.vl-settings .wrap { color: var(--muted) !important; }
/* ── SECTION LABELS ─────────────────────────────────────────── */
.vl-section-title {
  font-family: var(--font-h);
  font-size: 13px; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.vl-section-title::after {
  content: ''; flex: 1;
  height: 1px; background: var(--border);
}
/* ── FOOTER ─────────────────────────────────────────────────── */
#vl-footer {
  text-align: center;
  color: var(--muted); font-size: 12px;
  line-height: 1.75;
  margin-top: 36px; padding: 20px 24px;
  border-top: 1px solid var(--border);
}
#vl-footer .warn { color: #f87171; font-weight: 600; }
#vl-footer a { color: var(--accent); text-decoration: none; }
#vl-footer a:hover { text-decoration: underline; }
/* ── SCROLLBAR ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border2); }
/* ── ACCORDION (additional inputs) ─────────────────────────── */
.gradio-accordion {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.gradio-accordion .label-wrap span {
  color: var(--text) !important; font-family: var(--font-b) !important;
  font-weight: 500 !important;
}
"""

# ── 5. HTML Blocks ───────────────────────────────────────────────────────────
header_html = """
<div id="vitallm-header">
  <div class="vl-badge">
    <div class="vl-pulse"></div>
    Model Online &nbsp;·&nbsp; 50M Parameters
  </div>
  <h1>🏥 VitalLM-50M</h1>
  <p class="vl-sub">
    A specialized language model for clinical reasoning and medical assistance.
    Research use only — always consult a licensed physician.
  </p>
  <div class="vl-stats">
    <div class="vl-stat">
      <span class="vl-stat-num">50M</span>
      <span class="vl-stat-label">Parameters</span>
    </div>
    <div class="vl-stat">
      <span class="vl-stat-num">10</span>
      <span class="vl-stat-label">Layers</span>
    </div>
    <div class="vl-stat">
      <span class="vl-stat-num">256</span>
      <span class="vl-stat-label">Context</span>
    </div>
    <div class="vl-stat">
      <span class="vl-stat-num">BPE</span>
      <span class="vl-stat-label">Tokenizer</span>
    </div>
    <div class="vl-stat">
      <span class="vl-stat-num">Open</span>
      <span class="vl-stat-label">Source</span>
    </div>
  </div>
</div>
"""

footer_html = """
<div id="vl-footer">
  <span class="warn">⚠ Medical Disclaimer:</span>
  VitalLM-50M is an AI research model and is <strong>not a substitute for professional medical advice, diagnosis, or treatment.</strong><br>
  Always consult a qualified and licensed healthcare provider for medical decisions.<br><br>
  Built with ❤ using <a href="https://huggingface.co" target="_blank">Hugging Face Transformers</a> &amp;
  <a href="https://gradio.app" target="_blank">Gradio</a>.
</div>
"""

# ── 6. Build UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="VitalLM-50M · Medical Assistant") as demo:

    gr.HTML(header_html)

    with gr.Row(elem_classes="vl-main"):

        # ── Chat column ──────────────────────────────────────────
        with gr.Column(scale=3, elem_classes="vl-chat-col"):

            gr.HTML('<div class="vl-section-title">💬 Conversation</div>')

            chatbot = gr.Chatbot(
                label="",
                height=460,
                show_label=False,
                elem_id="vitallm-chatbot",
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=vitallm&backgroundColor=0f1628"),
            )

            with gr.Row(elem_id="vl-input-row"):
                txt = gr.Textbox(
                    placeholder="Describe your symptoms or ask a medical question…",
                    show_label=False,
                    scale=8,
                    lines=2,
                    max_lines=6,
                )
                with gr.Column(scale=1, min_width=90):
                    send_btn  = gr.Button("⬆ Send",  variant="primary",  size="sm", elem_id="send-btn")
                    clear_btn = gr.Button("✕ Clear", variant="secondary", size="sm", elem_id="clear-btn")

            gr.HTML('<div class="vl-section-title" style="margin-top:18px">💡 Quick Examples</div>')

            gr.Examples(
                examples=[
                    ["I have a persistent cough and fever. What could it be?"],
                    ["What are the early warning signs of Type 2 Diabetes?"],
                    ["How does hypertension affect the cardiovascular system?"],
                    ["What is the difference between a viral and bacterial infection?"],
                    ["Can you explain what a high white blood cell count may indicate?"],
                ],
                inputs=txt,
                label="",
            )

        # ── Settings column ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280, elem_classes=["vl-settings-col", "vl-settings"]):

            gr.HTML('<div class="vl-section-title">⚙ Model Settings</div>')

            system_msg = gr.Textbox(
                value=(
                    "You are a professional Medical Assistant with deep expertise in "
                    "clinical reasoning and evidence-based medicine. Provide clear, "
                    "accurate, and compassionate information. Always recommend consulting "
                    "a licensed physician for diagnosis and treatment."
                ),
                label="System Prompt",
                lines=6,
                info="Defines the model's role and behaviour.",
            )

            gr.HTML('<hr class="vl-divider">')
            gr.HTML('<div style="color:#64748b;font-size:12px;margin-bottom:10px;">Generation Parameters</div>')

            max_tokens = gr.Slider(
                minimum=1, maximum=200, value=100, step=1,
                label="Max New Tokens",
                info="Maximum tokens the model will generate.",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.4, step=0.1,
                label="Temperature",
                info="Higher = more creative. Lower = more focused.",
            )
            top_k = gr.Slider(
                minimum=1, maximum=100, value=40, step=1,
                label="Top-k Sampling",
                info="Number of top tokens considered at each step.",
            )

            gr.HTML("""
            <hr class="vl-divider">
            <div style="font-size:12px;color:#475569;line-height:1.6;">
              <strong style="color:#64748b;">Model architecture</strong><br>
              Layers: 10 &nbsp;·&nbsp; Heads: 8 &nbsp;·&nbsp; Embd: 512<br>
              Vocab: 16 384 &nbsp;·&nbsp; Context: 256
            </div>
            """)

    gr.HTML(footer_html)

    # ── Event wiring ─────────────────────────────────────────────
    def user_submit(user_msg, history):
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})
        return "", history

    def bot_reply(history, system_message, max_tokens, temperature, top_k):
        user_prompt = history[-2]["content"]
        
        for chunk in respond(
            user_prompt, history[:-2],
            system_message, max_tokens, temperature, top_k
        ):
            # Update the content of the final dictionary (the bot's response)
            history[-1]["content"] = chunk
            yield history

    txt.submit(
        user_submit, [txt, chatbot], [txt, chatbot]
    ).then(
        bot_reply, [chatbot, system_msg, max_tokens, temperature, top_k], chatbot
    )

    send_btn.click(
        user_submit, [txt, chatbot], [txt, chatbot]
    ).then(
        bot_reply, [chatbot, system_msg, max_tokens, temperature, top_k], chatbot
    )

    clear_btn.click(lambda: ([], ""), outputs=[chatbot, txt])


if __name__ == "__main__":
    demo.launch(css=custom_css, theme=gr.themes.Base())
