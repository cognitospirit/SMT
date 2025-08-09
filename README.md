# SMT
Self-Module Transformer to improve safety and better output

Progress Report: Development of the Self-Module Transformer (SMT)

Author: Navdeep Ahuja

Status: Stage 1 (Foundational Pre-training) in Progress

**1. Introduction**

This report details the initial progress in the development of the Self-Module Transformer (SMT), a novel language model architecture.

**1.1. Novel Architecture**
The SMT is a modified Transformer that includes a dedicated, trainable internal module called the "Self Area." Unlike standard models where a persona is an emergent and often inconsistent property, the Self Area is designed to learn a stable, abstract representation of a single, target persona. This module's output, a guidance_signal, is infused into every layer of the model's main processing body, ensuring the persona actively shapes the entire reasoning and generation process.

**1.2. Need for the Architecture**
The primary motivation for the SMT architecture is to create a more robust and reliable AI identity. By embedding the "self" or persona as a core architectural component, we aim to achieve two key objectives:

Increased Security: A hard-coded persona is theoretically more resistant to prompt injection and other adversarial attacks that attempt to make a model deviate from its intended character and safety guidelines.

Consistent, High-Quality Outputs: An intrinsic persona ensures that the model's tone, style, and behavior remain consistent across all interactions, leading to more predictable and trustworthy outputs.

**2. Methods**

A comprehensive plan was developed to build and train a proof-of-concept model from scratch, using feasible consumer-grade hardware. Gemini Pro-2.5 was used to brainstorm and write codes.

**2.1. Model & Data Specifications**

The following parameters were finalized for the initial build:

Model Architecture: A ~80 Million parameter SMT model.

Layers: 6

Embedding Dimension (d_model): 512

Attention Heads: 8

Feed-Forward Dimension: 2048

Self Area Dimension: 128

Training Dataset (Stage 1): A 10 GB text corpus derived from the English Wikipedia (wikimedia/wikipedia, 20231101.en snapshot), totaling approximately 2.5 billion tokens.

Training Dataset (Stage 2): A planned 500 MB synthetic dataset to be generated using a frontier model (e.g., Gemini 2.5 Pro) that perfectly embodies the target persona.

**3. Results to Date**

As of this report, the project is in the initial pre-training stage.

Training Progress: The SMT model has been successfully trained for 80,000 batches on the Wikipedia dataset using an NVIDIA RTX 4070 laptop GPU. The training process has been stable, with the loss value decreasing from a theoretical random-guessing baseline of ~10.4 to a consistent value of ~7.0.

Architectural Validation: An overfitting diagnostic test was performed. The model was trained repeatedly on a single batch of data, and the loss dropped significantly towards zero. This successful test confirms that the novel SMT architecture is fundamentally sound and capable of learning.

Current Model State: The model at 80,000 batches shows early, primitive signs of learning. It can replicate input and generates sequences of the most common English tokens (e.g., "the", ","). However, it has not yet learned complex grammar or semantics and is not yet capable of coherent text generation. Pre-training will continue.

**4. Discussion & Next Steps**

The initial results are promising and validate the core architectural concept. The primary challenge observed is the significant time investment required for pre-training a model from scratch, even at a small scale.

For future, faster iterations of this research, a more efficient approach could be explored. Instead of training from zero, one could perform "model surgery" on a powerful, existing open-source pre-trained model (such as Microsoft's Phi-3 or Meta's Llama 3 8B). This would involve programmatically inserting the Self Area module into the pre-trained architecture. The model would then only need to undergo the much faster Stage 2 fine-tuning process, leveraging the advanced foundational training already completed while still allowing for the testing of the novel self-guidance mechanism.

The immediate plan is to continue the current pre-training run to at least 250,000 batches to observe the emergence of more complex language structures.
