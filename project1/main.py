import streamlit as st
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource
def load_model():
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


st.set_page_config(page_title="LLM Playground", layout="wide")
st.title("The tokenizer viewer")
st.markdown(
    "LLMs don't read words. The read **Integers**. Let's see how they translate."
)

with st.sidebar:
    model_encoding = st.selectbox(
        "Choose tokenizer encoding",
        options=[
            "cl100k_base",
            "p50k_base",
            "r50k_base",
        ],
        index=0,
        help="cl100k_base is used by GPT-4. p50k is GPT-3. r50k is GPT-2.",
    )
    st.divider()
    temperature = st.slider(
        "Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1
    )
    top_k = st.slider("Top-K", min_value=1, max_value=100, value=50)
    top_p = st.slider(
        "Top-P (Nucleus)", min_value=0.0, max_value=1.0, value=0.9, step=0.01
    )
    st.divider()
    max_length = st.slider("Max new tokens", 1, 100, 20)
    generate_button = st.button("Generate story")

text = st.text_area("Type something here:", "The capital of France is ", height=150)


def apply_top_k(logits):
    if top_k < logits.shape[-1]:
        top_k_cutoff = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < top_k_cutoff] = float("-inf")


def apply_top_p(logits):
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        logits[mask] = float("-inf")


if text:
    enc = tiktoken.get_encoding(model_encoding)
    token_ids = enc.encode(text)
    token_strings = [enc.decode([token_id]) for token_id in token_ids]
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", len(text))
    col2.metric("Tokens", len(token_ids))

    ratio = len(text) / len(token_ids) if len(token_ids) > 0 else 0
    col3.metric("Compression Ratio", f"{ratio:.2f} chars/token")

    st.subheader("The Translation")

    html_string = ""
    colors = ["#FFDDC1", "#C1E1C1", "#C1D4E1", "#E1C1D4", "#E1E1C1"]

    for i, (t_str, t_id) in enumerate(zip(token_strings, token_ids)):
        color = colors[i % len(colors)]
        t_str_safe = t_str.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "⏎")

        span = f'<span style="background-color:{color}; padding: 0 4px; color: black; border-radius: 4px; margin: 0 2px; display: inline-block; border: 1px solid #ccc;" title="ID: {t_id}">{t_str_safe}</span>'
        html_string += span

    st.markdown(html_string, unsafe_allow_html=True)

    st.divider()
    st.subheader("The Raw Input Tensor")
    st.code(str(token_ids), language="json")

st.divider()
st.subheader("Model Prediction")

if text:
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]

    # Divide by temperature, values below 1 will make the probability differences
    # between answers to be larger, while above 1 will make them smaller, answers become
    # equally probable to be chosen
    scaled_logits = next_token_logits / temperature

    apply_top_k(scaled_logits)

    apply_top_p(scaled_logits)

    probs = torch.softmax(scaled_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, top_k)

    st.write(f"The model thinks the next word after '{text.split()[-1]}' might be:")

    for i in range(len(top_probs)):
        probability = top_probs[i].item() * 100

        if probability > 0.00:
            token_name = repr(tokenizer.decode([top_ids[i]]))
            st.write(f"ID: {top_ids[i]} | **'{token_name}'** ({probability:.2f}%)")

if generate_button and text:
    st.subheader("Generated output")

    input_ids = tokenizer.encode(text, return_tensors="pt")
    current_ids = input_ids

    output_placeholder = st.empty()
    generated_text = text

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(current_ids)
            next_token_logits = outputs.logits[0, -1, :]
            scaled_logits = next_token_logits / temperature

            apply_top_k(scaled_logits)

            apply_top_p(scaled_logits)

            probs = torch.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=-1)
            new_word = tokenizer.decode(next_token_id[0])
            generated_text += new_word

            output_placeholder.markdown(f"{generated_text}")

            if next_token_id.item() == tokenizer.eos_token_id:
                break
