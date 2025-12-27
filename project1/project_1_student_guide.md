# Project 1 Student Guide: The LLM Playground 🛝
**Instructor:** Ali Aminian

## 👋 Introduction
We begin at the beginning: **Pre-Training**. In Project 1, we map the entire lifecycle of an LLM, from raw internet noise to a polished Assistant. We will touch on the specific datasets and algorithms used in modern state-of-the-art models like Llama and GPT.

## 🎯 Learning Objectives
1.  **Master Data Engineering:** Understand the pipelines (FineWeb, Dolma) that clean Common Crawl data.
2.  **Dissect Architectures:** Deep dive into Neural Networks, Transformers, and GPT vs Llama families.
3.  **Implement Decoding:** Greedy, Beam Search, Top-K, Top-P.
4.  **Align Models:** Understand Reinforcement Learning (RL) and RLHF (PPO).

## 🛠️ Tech Stack
*   **Python / Streamlit**
*   **TikToken** (BPE Tokenization)
*   **PyTorch / Transformers**
*   **Datasets** (Hugging Face)
*   **Trafilatura** (Text Extraction)

---

## 🗺️ Implementation Roadmap

### 🎓 Phase 0: Theory - LLM Overview and Foundations
**Concept:** The 3 stages of an LLM's life.
1.  **Pre-Training (Base Model):** Learning patterns from the raw internet (Neural Network foundations).
2.  **SFT (Instruction Tuning):** Learning to follow user commands.
3.  **RL and RLHF:** Aligning the model with human values and safety.

---

### Phase 1: Pre-Training Data (The "FineWeb" Pipeline)
**🎓 Theory: Garbage In, Garbage Out**
*   **Why clean?** If a model reads the same viral tweet 10,000 times (Duplicates), it overfits/memorizes instead of learning logic.
*   **Sources:** Manual Crawling vs. **Common Crawl**.
*   **Cleaning:** The secret sauce of models like Falcon (**RefinedWeb**) and Olmo (**Dolma**).

**Goal:** Clean the internet.
*   **Task:** Build a "Data Cleaner" widget.
    *   Input: Raw HTML text with noise.
    *   Pipeline: 
        1. **Trafilatura:** Extract main text from HTML.
        2. **Filtering:** Remove short lines/Ads.
        3. **PII Removal:** Regex masking (Emails/Phones).
        4. **Deduplication:** MinHash (Concept).
    *   Output: Clean "FineWeb" style text.

### Phase 2: Tokenization & Architecture
**Goal:** Turn text into tensors.

*   **Step 2.1:** **Tokenization.**
    *   Implement **BPE (Byte Pair Encoding)**.
    *   Visualize the split.
*   **Step 2.2:** **Architecture Inspector (Transformers).**
    *   Visualize the difference between **GPT Family** (OpenAI) and **Llama Family** (Meta).
    *   *Highlight:* 
        *   **Positional Embeddings:** Learned (GPT-2) vs **RoPE** (Llama).
        *   **Normalization:** LayerNorm (Post-Norm) vs **RMSNorm** (Pre-Norm).

### Phase 3: Text Generation (Inference)
**Goal:** The `generate()` function.

*   **Task:** Implement 4 decoding strategies side-by-side:
    1.  **Greedy:** Determinstic.
    2.  **Beam Search:** Exploring multiple paths (Breadth-first).
    3.  **Top-K:** Truncating the tail.
    4.  **Top-P (Nucleus):** Dynamic cumulative probability.

### Phase 4: Post-Training (The Alignment)
**Goal:** SFT and RLHF.

*   **Step 4.1:** **SFT (Supervised Fine-Tuning).**
    *   Train on "Instruction-Response" pairs.
*   **Step 4.2:** **Reinforcement Learning (RL) and RLHF.**
    *   **Reward Modeling:** Train a model to score answers (Thumbs up/down).
    *   **PPO (Proximal Policy Optimization):** Use the Reward Model to update the LLM's policy.
    *   **Step 4.3: Verifiable Tasks (RLVR).**
        *   *Concept:* Using objective tasks (Math/Code) to train reasoning models (DeepSeek-R1 style).
    *   *Simulate:* Show a "Reward Score" changing as the response gets better.

### Phase 5: Evaluation
**Goal:** Measuring success.

*   **Metrics:**
    *   **Traditional Metrics:** Perplexity.
    *   **Task-Specific Benchmarks:** MMLU, GSM8K.
    *   **Human Evaluation:** Elo ratings (Chatbot Arena) and Leaderboards.

### Phase 6: Chatbots' Overall Design
**Goal:** The System View.

*   **The Full Stack:**
    *   **Context Management:** Handling history windows.
    *   **System Prompts:** Defining the "Persona".
    *   **Safety Layers:** Guardrails against toxicity.
