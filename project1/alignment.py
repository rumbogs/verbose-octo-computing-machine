import json

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource
def load_model():
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    return tokenizer, model


tab1, tab2, tab3 = st.tabs(["SFT", "Reward Modeling", "RLHF (PPO/DPO)"])


if "preferences" not in st.session_state:
    st.session_state.preferences = []

with tab1:
    st.subheader("Supervised Fine-Tuning (SFT)")
    data = []
    with open("sft_dataset.json") as f:
        data = json.load(f)
        st.write("Example training data")
        st.json(data[0:5])

    train_button = st.button("Train")
    reset_button = st.button("Reset model")

    if reset_button:
        st.cache_resource.clear()
        st.rerun()

    epochs = 5

    tokenizer, model = load_model()

    if train_button:
        adamw_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()

        placeholder = st.empty()
        loss_items = []

        for i in range(epochs):
            for item in data:
                prompt = (
                    f"Instruction:{item['instruction']}\nResponse: {item['response']}"
                )
                inputs = tokenizer(prompt, return_tensors="pt")
                model_output = model(**inputs, labels=inputs["input_ids"])
                loss = model_output.loss
                loss_items.append(loss.item())
                loss.backward()
                adamw_optimizer.step()
                adamw_optimizer.zero_grad()
                placeholder.line_chart(loss_items)

        model.eval()

    text = st.text_area("Test Prompt", "Who are you?", height=150)

    if text:
        inputs = tokenizer(f"Instruction: {text}\nResponse: ", return_tensors="pt")
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        st.subheader("Generated")
        st.markdown(decoded)


with tab2:
    st.subheader("Human Labeler")
    text = st.text_area(
        "Prompt:", "The extraordinary gentlemen bent down and", height=150
    )

    generate_button = st.button("Generate candidates")

    col1, col2 = st.columns(2)

    tokenizer, model = load_model()

    outputA = None
    outputB = None

    if generate_button and text:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputA = model.generate(
                **inputs,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                max_new_tokens=50,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputB = model.generate(
                **inputs,
                temperature=0.1,
                top_k=1,
                do_sample=True,
                max_new_tokens=50,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        decodedA = tokenizer.batch_decode(outputA, skip_special_tokens=True)[0]
        decodedB = tokenizer.batch_decode(outputB, skip_special_tokens=True)[0]

        with col1:
            st.subheader("Answer A")
            st.markdown(decodedA)
            acceptA = st.button("A is better")

        with col2:
            st.subheader("Answer B")
            st.markdown(decodedB)
            acceptB = st.button("B is better")

        if acceptA and outputA and outputB:
            st.session_state.preferences.append(
                {"prompt": text, "chosen": decodedA, "rejected": decodedB}
            )

        if acceptB and outputA and outputB:
            st.session_state.preferences.append(
                {"prompt": text, "chosen": decodedB, "rejected": decodedA}
            )

with tab3:
    st.subheader("N/A")
