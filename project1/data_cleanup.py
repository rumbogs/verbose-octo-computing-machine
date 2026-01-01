import re
from random import randint

import streamlit as st
import trafilatura

st.set_page_config(page_title="Data Cleanup", layout="wide")
st.title("Data cleanup")

num_permutations = 100

max_val = (2**32) - 1
prime = 4294967311
random_coef = [
    (randint(0, max_val), randint(0, max_val)) for _ in range(num_permutations)
]

with st.sidebar:
    apply_heuristics = st.checkbox("Apply heuristics", value=False, key="clean_html")
    demo_mode = st.checkbox("Similarity demo mode", value=False, key="demo_mode")


def filter_text(text):
    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        line = line.strip()
        print(line)
        if len(line) <= 40:
            continue
        if not line.endswith((".", "!", "?", '"', "'", "]")):
            continue
        filtered_lines.append(line)
    return "\n\n".join(filtered_lines)


def clean_pii(text):
    cleaned_text = re.sub(r"\S+@\S+\.\S+", "||EMAIL||", text)
    cleaned_text = re.sub(
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "||IP_ADDRESS||", cleaned_text
    )
    return cleaned_text


def get_minhash_signature(text, shingle_size=5):
    signature = [float("inf")] * num_permutations
    for i in range(num_permutations):
        split_text = [
            text[i : i + shingle_size] for i in range(len(text) - shingle_size + 1)
        ]
        hash_text = [
            (hash(shingle) * random_coef[i][0] + random_coef[i][1]) % prime
            for shingle in split_text
        ]
        min_val = min(hash_text)
        signature[i] = min_val

    return signature


def compare_signatures(a, b):
    same = 0
    # This assumes same lengths of signatures
    for i, val in enumerate(a):
        # st.write(f"a {val}")
        # st.write(f"b {b[i]}")
        if val == b[i]:
            same += 1

    return same / len(a)


def extract_text_from_url(url):
    html_text = trafilatura.fetch_url(url)
    cleaned_text = trafilatura.extract(html_text)
    cleaned_text = clean_pii(cleaned_text)

    if apply_heuristics:
        cleaned_text = filter_text(cleaned_text)

    return cleaned_text


if not demo_mode:
    url = st.text_input(
        "URL to clean",
        value="https://en.wikipedia.org/wiki/World_Clock_(Alexanderplatz)",
    )
    url_2 = st.text_input(
        "URL to diff", value="https://en.wikipedia.org/wiki/Email_address"
    )

    cleaned_text = extract_text_from_url(url)
    cleaned_text_2 = extract_text_from_url(url_2)

    minhash_1 = get_minhash_signature(cleaned_text)
    minhash_2 = get_minhash_signature(cleaned_text_2)

    st.divider()
    st.write(
        f"The two texts are {compare_signatures(minhash_1, minhash_2) * 100}% similar."
    )

    st.divider()
    st.subheader("Output")

    st.write(cleaned_text)
else:
    text1 = st.text_area(
        "Text A", "The quick brown fox jumps over the lazy dog.", height=150
    )
    text2 = st.text_area(
        "Text B", "The quick brown fox jumps over the lazy cat.", height=150
    )

    minhash_1 = get_minhash_signature(text1)
    minhash_2 = get_minhash_signature(text2)

    st.divider()
    st.write(
        f"The two texts are {compare_signatures(minhash_1, minhash_2) * 100}% similar."
    )
