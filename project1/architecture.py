import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tiktoken
from sympy.core.random import rng
from transformers import AutoConfig

MAX_POSITIONS = 20


@st.cache_resource
def get_configs():
    c1 = AutoConfig.from_pretrained("gpt2")
    c2 = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")
    return c1, c2


st.title("Architecture Inspector")

tab1, tab2, tab3 = st.tabs(
    ["Specs comparison", "Positional Embeddings (RoPE)", "Tokenizer Efficiency"]
)

with tab1:
    st.header("Legacy (GPT-2) vs Modern (Llama)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Legacy (GPT-2)")
        config_gpt, _ = get_configs()
        st.write(f"**Vocab Size:** {config_gpt.vocab_size}")
        st.write(f"**Positional Embeddings:** Learned ({config_gpt.n_positions})")
        st.write("**Norm:** LayerNorm")

    with col2:
        st.subheader("Modern (SmolLM)")
        _, config_gpt = get_configs()
        st.write(f"**Vocab Size:** {config_gpt.vocab_size}")
        st.write("**Positional Embeddings:** RoPE")  # Check rope_theta
        st.write("**Norm:** RMSNorm")

with tab2:
    st.header("Understanding RoPE")

    m = st.slider("Query Position", min_value=0, max_value=MAX_POSITIONS, value=0)
    n = st.slider("Key Position", min_value=0, max_value=MAX_POSITIONS, value=0)

    st.divider()

    base_angle = np.pi / 10  # 18 degrees

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Legacy")

        st.latex(r"P_{pos} = \text{EmbeddingTable}[pos]")
        st.markdown(
            "The model treats every position as a unique, independent ID. It must memorize what 'Position 5' looks like. It cannot generalize to 'Position 500' if it has never seen it during training."
        )

        rng = np.random.RandomState(42)
        z_vals = np.arange(MAX_POSITIONS + 1)
        x_legacy = rng.uniform(-1, 1, size=MAX_POSITIONS + 1)
        y_legacy = rng.uniform(-1, 1, size=MAX_POSITIONS + 1)

        x_plot, y_plot, z_plot = [], [], []
        for i in range(MAX_POSITIONS + 1):
            x_plot.extend([0, x_legacy[i], None])
            y_plot.extend([0, y_legacy[i], None])
            z_plot.extend([z_vals[i], z_vals[i], None])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="lines+markers",
                name="Learned Embeddings",
                line=dict(color="orange"),
            )
        )

        st.plotly_chart(fig)

    with col2:
        st.subheader("Interactive RoPE Visualiser")

        st.latex(r"P_{pos} = \text{Rotate}(x, y, \theta \cdot pos)")
        st.markdown(
            "The model treats position as a continuous rotation. Position 6 is just Position 5 rotated by a fixed angle. This allows the model to understand relative distances and extrapolate to much longer sequences than it was trained on."
        )

        z_all = np.linspace(0, MAX_POSITIONS, 100)
        theta_all = z_all * base_angle
        x_all = np.cos(theta_all)
        y_all = np.sin(theta_all)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x_all,
                y=y_all,
                z=z_all,
                mode="lines",
                name="RoPE Manifold",
                line=dict(color="lightgray"),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[0, np.cos(m * base_angle)],
                y=[0, np.sin(m * base_angle)],
                z=[m, m],
                mode="lines+markers",
                name="Query",
                line=dict(color="blue", width=5),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[0, np.cos(n * base_angle)],
                y=[0, np.sin(n * base_angle)],
                z=[n, n],
                mode="lines+markers",
                name="Key",
                line=dict(color="red", width=5),
            )
        )

        st.plotly_chart(fig)


def render_token_viz(tokens, token_strings):
    st.metric("Characters", len(text))
    st.metric("Tokens", len(tokens))

    ratio = len(text) / len(tokens) if len(tokens) > 0 else 0
    st.metric("Compression Ratio", f"{ratio:.2f} chars/token")

    st.subheader("The Translation")

    html_string = ""
    colors = ["#FFDDC1", "#C1E1C1", "#C1D4E1", "#E1C1D4", "#E1E1C1"]

    for i, (t_str, t_id) in enumerate(zip(token_strings, tokens)):
        color = colors[i % len(colors)]
        t_str_safe = t_str.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "⏎")

        span = f'<span style="background-color:{color}; padding: 0 4px; color: black; border-radius: 4px; margin: 0 2px; display: inline-block; border: 1px solid #ccc;" title="ID: {t_id}">{t_str_safe}</span>'
        html_string += span

    st.markdown(html_string, unsafe_allow_html=True)

    st.divider()
    st.subheader("The Raw Input Tensor")
    st.code(str(tokens), language="json")


with tab3:
    st.header("Vocabulary Impact")
    st.markdown("Larger vocabulary = Fewer tokens = Faster generation.")

    st.divider()

    text = st.text_area("Type something here:", "The capital of France is", height=150)

    if text:
        legacy_encoder = tiktoken.get_encoding("r50k_base")
        modern_encoder = tiktoken.get_encoding("cl100k_base")

        l_tokens = legacy_encoder.encode(text)
        m_tokens = modern_encoder.encode(text)

        l_token_strings = [legacy_encoder.decode([token_id]) for token_id in l_tokens]
        m_token_strings = [modern_encoder.decode([token_id]) for token_id in m_tokens]

        l_len = len(l_tokens)
        m_len = len(m_tokens)

        improvement = ((l_len - m_len) / l_len) * 100 if l_len > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("GPT-2 Tokens", l_len)
        col2.metric("GPT-4 Tokens", m_len)
        col3.metric(
            "Efficiency Gain",
            f"{improvement:.1f}%",
            delta=f"{l_len - m_len} tokens",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("GPT-2")

            render_token_viz(l_tokens, l_token_strings)

        with col2:
            st.subheader("GPT-4")

            render_token_viz(m_tokens, m_token_strings)
