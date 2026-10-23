1. What is LLaMA?
LLaMA stands for Large Language Model Meta AI. It is a family of large language models developed by Meta AI.

LLaMA models are based on the Transformer decoder-only architecture. They generate text by predicting the next token based on the tokens that came before it.

For example, suppose the input is:
"Retrieval-Augmented Generation uses"

The model might predict the next token as external, followed by knowledge, and continue generating the answer one token at a time.









Model family: LLaMA / Llama

Architecture: Decoder-only Transformer

Main task: Next-token prediction

Common applications: Chatbots, RAG, summarization, coding assistants, document Q&A, and agentic applications.



Why is LLaMA important for GenAI interviews?
Interviewers want to know whether you understand how an LLM works internally, rather than just how to call an API.

You should be able to explain:
- How text becomes tokens and embeddings.
- How attention determines which tokens matter.
- How RoPE provides positional information.
- How SwiGLU and RMSNorm work.
- How the model predicts the next token.
- How instruction tuning and alignment turn a base model into a chat assistant.
- How quantization, KV caching, and fine-tuning support real-world applications.
2. Complete LLaMA architecture
The easiest way to understand LLaMA is to follow the path from input text to generated output.
1. Input text
"Explain RAG architecture"

2. Tokenizer

Converts text into token IDs

3. Token embeddings

Converts token IDs into numerical vectors

4. Repeated Transformer decoder blocks
RMSNorm

Causal self-attention + RoPE

Q, K, V and attention heads

Residual connection
RMSNorm → SwiGLU feed-forward network

Residual connection

Repeat across many layers

5. Final RMSNorm + linear output layer

Produces a score (logit) for each vocabulary token

6. Next-token selection

Selects a token, appends it, and repeats


This diagram describes the common text-only LLaMA design used by earlier generations. Specific components vary across model generations. For example, Llama 4 introduces a mixture-of-experts architecture and native multimodal capabilities. 

GitHub
+2



3. Key components explained in detail
3.1 Tokenizer
The tokenizer converts text into smaller units called tokens. A token can represent a word, part of a word, punctuation, or another text unit.

Example:
Input: "I love GenAI"

Possible tokens:
["I", " love", " Gen", "AI"]

Token IDs:
[101, 248, 7361, 419]  # Illustrative IDs only


The actual tokenization and token IDs depend on the tokenizer used by the specific model.

Why is tokenization needed?

Neural networks operate on numbers, not raw text. Tokenization maps the input into vocabulary IDs that can be processed by the model.

Interview points:
- Token IDs are integers, not embeddings.
- The tokenizer vocabulary defines which token IDs are available.
- Subword tokenization helps represent uncommon words without needing a separate token for every possible word.
- Special tokens can mark message boundaries, end-of-turn signals, or other control information.
Tricky question: Does one token always equal one word?

Answer: No. A word can be one token or several tokens. A token can also be punctuation, whitespace-associated text, or part of a word.

3.2 Token embeddings
Token IDs are converted into dense numerical vectors through a learned embedding matrix.

Suppose the embedding dimension is 4 for a simplified example:
Token	Token ID	Example embedding
I	101	[0.2, 0.4, 0.1, 0.8]
love	248	[0.5, 0.3, 0.9, 0.2]
AI	419	[0.7, 0.1, 0.6, 0.4]
These numbers are illustrative, not real LLaMA weights.

If the input has 3 tokens and the embedding dimension is 4, the embedding tensor has shape:
\[
X \in \mathbb{R}^{3 \times 4}
\]

In a real model, the embedding dimension is much larger.

Important distinction:
- Token ID: a discrete integer.
- Embedding: a learned vector.
- Hidden state: a vector that gets updated as it passes through Transformer layers.
The embedding initially represents learned token features. After attention and feed-forward processing, hidden states contain contextual information about the input.

3.3 Transformer decoder blocks
This is the core of LLaMA.

A decoder block has two major computational sublayers:
1. Causal self-attention.
2. A feed-forward network, commonly using SwiGLU.
It also includes RMSNorm and residual connections.

In the classic LLaMA architecture, the computation is arranged in a pre-normalization pattern:
\[
h = x + \operatorname{Attention}(\operatorname{RMSNorm}(x))
\]

\[
y = h + \operatorname{FFN}(\operatorname{RMSNorm}(h))
\]

Here, \(x\) is the incoming hidden state, \(h\) is the state after attention and its residual connection, and \(y\) is the block output. 

GitHub
+1




Let's understand each part.
3.4 Self-attention: how tokens use context
Self-attention allows each token to gather relevant information from earlier tokens.

Consider:
"The developer fixed the bug because she understood the issue."

The model needs to connect "she" with the relevant person and relate "the issue" to the surrounding context. Attention provides a mechanism for learning such relationships.

The attention computation uses three vectors:
- Query (Q): What information is this token looking for?
- Key (K): What information does each token make available for matching?
- Value (V): What information does each token contribute?
These are learned projections of the hidden states.
\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

The matrices \(W_Q\), \(W_K\), and \(W_V\) are learned during training.
How attention works
1. Generate Q, K, and V vectors for the input tokens.
2. Calculate query-key similarity scores to determine which tokens are relevant.
3. Scale the scores and apply a causal mask so a token cannot attend to future tokens.
4. Apply softmax to convert scores into attention weights.
5. Use the weights to combine the value vectors into contextual representations.
The standard scaled dot-product attention equation is:
\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

Where:
- \(QK^T\) computes query-key similarities.
- \(d_k\) is the key dimension per attention head.
- \(M\) is the causal mask, where future positions receive a very negative score.
- Softmax converts the scores into weights.
- Multiplication by \(V\) combines information from the tokens.
Why divide by \(\sqrt{d_k}\)?

Dot products can become large as the key dimension increases. Scaling helps keep the softmax scores in a more manageable range and supports stable learning.
What is causal attention?
In a decoder-only LLM, a token can use earlier tokens but must not use future tokens when predicting the next token.

For a sequence of four tokens, the allowed attention pattern looks like this:
Rows are the token doing the attending; columns are the token being attended to.
T1


T2


T3


T4




































Allowed


Blocked future tokens



Interview question: Why does LLaMA use decoder-only architecture instead of an encoder-decoder architecture?

The decoder-only design is well suited to autoregressive text generation: the model learns to predict the next token from preceding context. An encoder-decoder model instead typically uses a separate encoder to process an input and a decoder to generate an output, often using cross-attention between them.

This does not mean encoder-decoder models are inferior; they are useful for tasks such as translation and sequence-to-sequence generation.
3.5 Multi-head attention and Grouped-Query Attention
Rather than calculating attention only once, multi-head attention uses multiple heads. Different heads can learn different relationships in the sequence.

For example, one head might learn useful syntactic relationships, while another learns long-distance contextual relationships. These are learned specializations, not fixed assignments.

Classic multi-head attention can be expensive because each head maintains its own keys and values.

Grouped-Query Attention (GQA) reduces this cost by sharing key and value heads across groups of query heads.
GQA example
8 query heads

Q1, Q2
KV1



Q3, Q4
KV2



Q5, Q6
KV3



Q7, Q8
KV4




Illustration: 2 query heads share each KV head.


In this example, 8 query heads share 4 key/value heads. This reduces the amount of key/value state that must be stored and can improve inference efficiency, especially when generating long sequences.

Know the distinction:
Attention type	Query heads	Key/value heads
MHA — Multi-Head Attention	Multiple	Same number as query heads
MQA — Multi-Query Attention	Multiple	One shared KV head
GQA — Grouped-Query Attention	Multiple	Fewer than query heads, more than one
GQA is used in Llama 2 70B, Llama 3, and later variants, but the exact configuration depends on the model. 

GitHub
+2



3.6 RoPE — Rotary Positional Embeddings
Attention needs information about token order. Otherwise, it would have difficulty distinguishing different arrangements of the same tokens.

For example:
- "The dog chased the cat."
- "The cat chased the dog."
RoPE stands for Rotary Positional Embeddings. It injects position-dependent rotations into the query and key representations, allowing attention scores to incorporate relative positional relationships.

Conceptually:
\[
Q'=R_m Q,\qquad K'=R_n K
\]

Here, \(R_m\) and \(R_n\) are position-dependent rotations for token positions \(m\) and \(n\).

The resulting query-key similarity depends on their relative position as well as their content.

Why does LLaMA use RoPE?
- It provides positional information without simply adding a learned positional vector to each token embedding.
- It supports useful relative-position behavior in attention.
- RoPE scaling techniques can help certain model variants support longer contexts.
Interview trap: Does RoPE alone guarantee that a model can handle an unlimited context length?

No. Context length depends on the model configuration, training, positional scaling, memory requirements, and implementation. A model's advertised context limit should not be assumed to be unlimited.
3.7 RMSNorm — Root Mean Square Normalization
Normalization helps keep hidden-state magnitudes controlled during training and inference.

LLaMA uses RMSNorm in place of the original Transformer architecture's commonly used LayerNorm.

A simplified RMSNorm formula is:
\[
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot g
\]

Where:
- \(x\) is the input vector.
- \(\epsilon\) is a small constant for numerical stability.
- \(g\) is a learned scaling vector.
- \(\odot\) represents element-wise multiplication.
RMSNorm vs LayerNorm
Feature	RMSNorm	LayerNorm
Normalizes using	Root mean square	Mean and variance
Explicit mean subtraction	No	Yes
Learnable scaling	Commonly yes	Commonly yes
Learnable bias	Not required in the classic LLaMA implementation	Often present
Main motivation	Simpler normalization computation	Center and scale activations
RMSNorm does not subtract the mean. It normalizes the magnitude of the input using its root mean square.
3.8 SwiGLU — Feed-forward network
After attention, each token representation passes through a feed-forward network (FFN).

Attention mixes information across tokens. The FFN applies learned transformations to each token's representation independently, adding nonlinear processing capacity.

Classic LLaMA uses a SwiGLU-style FFN.

A simplified expression is:
\[
\operatorname{FFN}(x)
=W_2\left[\operatorname{SiLU}(W_1x)\odot W_3x\right]
\]

Here:
- \(W_1\) and \(W_3\) produce intermediate representations.
- SiLU is an activation function.
- The element-wise product acts as a gate.
- \(W_2\) projects the gated representation back to the model's hidden dimension.
What is SiLU?
\[
\operatorname{SiLU}(x)=x\cdot \operatorname{sigmoid}(x)
\]

Unlike ReLU, SiLU is smooth and can produce small negative values.

Interview question: Why use SwiGLU instead of a simple ReLU FFN?

SwiGLU uses a gating mechanism to control information flowing through the intermediate representation. It can provide effective model capacity and training performance, although its benefits depend on the architecture and training setup.
3.9 Residual connections
A residual connection adds a sublayer's output to its input.
\[
y=x+F(x)
\]

In LLaMA, attention and the feed-forward network each have residual connections.

Why are they useful?
- They help preserve information from earlier representations.
- They support gradient flow through deep networks.
- They make it easier for a sublayer to learn a refinement to its input.
Interview trap: Does the residual connection replace the attention or FFN output?

No. It adds the sublayer output to the incoming hidden state.
3.10 Final linear layer and logits
After the final decoder block and final normalization, a linear layer maps the hidden representation to a vector of vocabulary scores.

Suppose the vocabulary contains 50,000 tokens. The output for one position can contain 50,000 logits.
Candidate token	Illustrative logit
retrieval	4.8
database	3.1
model	2.7
banana	-0.6
A logit is a raw score, not a probability.

Softmax can convert logits into probabilities:
\[
P(t_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

The model then uses a decoding strategy to choose the next token. The highest-probability token is not always chosen; temperature, top-k, top-p, and other decoding settings can influence selection.

The chosen token is appended to the sequence, and generation continues until a stopping condition is reached.
4. How LLaMA works end to end
Let's use a simple prompt:
Explain RAG

Step 1: Tokenization
The tokenizer converts the prompt into token IDs.
prompt = "Explain RAG"# Conceptual representationtoken_ids = [token_1, token_2, token_3]



These names are placeholders; actual token IDs come from the model's tokenizer.
Step 2: Embedding lookup
The model looks up a vector for each token ID.
Token IDs
   ↓
Embedding matrix
   ↓
Numerical vectors


The vectors are the input hidden states for the first Transformer block.
Step 3: Processing through decoder layers
Each layer applies attention, normalization, residual connections, and the feed-forward network.

The hidden representation becomes increasingly contextual as it passes through the layers.

For example, the model can learn that "RAG" in this context likely refers to Retrieval-Augmented Generation.
Step 4: Predicting the next token
The final output layer produces vocabulary logits. The decoding algorithm selects a token such as stands or is, depending on the learned distribution and decoding settings.
Step 5: Autoregressive generation
The selected token is added to the input, and the model predicts another token.
Prompt: Explain RAG
         ↓
Predict: RAG
         ↓
Append generated token
         ↓
Predict next token
         ↓
Repeat
         ↓
Complete answer


The generated answer is not normally produced all at once. It is generated token by token, although implementation optimizations can make the process much more efficient.
Step 6: KV cache optimization
During generation, the model can cache the keys and values calculated for previous tokens.

Without a cache, it would repeatedly recompute much of the previous context's attention information. With a KV cache, it can reuse previously calculated keys and values and compute the new token's representations.

Interview-ready answer:

"During prefill, the model processes the prompt and builds the key-value cache. During decoding, it processes the newly generated token, reuses the cached keys and values, computes the next-token distribution, and repeats until a stopping condition is reached."

The cache consumes memory, which is one reason long-context inference can be expensive. GQA can reduce this memory requirement by reducing the number of KV heads. 

GitHub
+1



5. How LLaMA is trained
Understanding training is essential for GenAI interviews because a model's architecture and its training process solve different problems.

There are three major stages to understand.

Stage 1 — Pretraining

The model learns language patterns, facts, code, and other statistical relationships from large training datasets. Its central objective is next-token prediction.




Stage 2 — Supervised fine-tuning (SFT)

The model learns from example prompts and desired responses, helping it follow instructions and answer in a conversational format.




Stage 3 — Preference and alignment training

Methods such as RLHF or DPO can further train the model to prefer responses that align with human or curated preferences. The exact recipe depends on the model.



5.1 Next-token prediction and loss
During pretraining, the model receives a sequence and learns to predict the next token at each position.

For example:
Input context	Target token
Python is	a
Python is a	programming
Python is a programming	language
These are illustrative training pairs. Actual training predicts targets over sequences of tokens.

A common objective is cross-entropy loss:
\[
\mathcal{L}=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})
\]

Here:
- \(x_t\) is the target token.
- \(x_{<t}\) is the preceding context.
- \(P_\theta\) is the model's predicted probability.
- \(\mathcal{L}\) measures how poorly the model predicts the target tokens.
The training algorithm uses this loss to update model parameters through backpropagation and gradient-based optimization.
5.2 Base model vs instruction-tuned model
Base LLaMA model	Instruction-tuned LLaMA model
Primarily trained to predict the next token	Further trained to follow instructions
Good at continuing text	Better suited to conversational interactions
May continue a prompt in unexpected formats	More likely to produce structured answers
Not necessarily optimized for assistant behavior	Tuned for assistant-style responses
Important: Instruction tuning does not automatically make a model factually correct, immune to hallucinations, or secure against every prompt injection.
5.3 Fine-tuning with LoRA and QLoRA
If an organization wants a model to follow a specialized format or perform a domain-specific task, it might fine-tune an existing model rather than train one from scratch.
- Full fine-tuning: Updates the model's trainable parameters.
- LoRA: Adds small trainable low-rank adapter matrices while keeping the original weights frozen.
- QLoRA: Uses a quantized base model with trainable LoRA adapters, reducing memory requirements.
For many domain-specific experiments, LoRA or QLoRA is more practical than full fine-tuning.
6. LLaMA generations: what changes?
Do not assume every model with the LLaMA name has exactly the same architecture.
Generation	Key interview takeaway
LLaMA 1	Decoder-only Transformer with RMSNorm, RoPE, and SwiGLU
Llama 2	Retains the core architecture; model sizes and configurations vary, and GQA is used in the 70B model
Llama 3 / 3.1 / 3.2 / 3.3	Builds on the decoder-only design; uses GQA in relevant text models, with different sizes, context capabilities, and modalities across variants
Llama 4	Introduces a mixture-of-experts (MoE) design and native multimodal capabilities
The exact number of layers, hidden dimensions, KV heads, vocabulary size, context length, and other settings depend on the particular checkpoint. See the LLaMA model implementation and Hugging Face Llama documentation. 

GitHub
+2



What is Mixture of Experts (MoE)?
Llama 4 adds a major architectural difference.

Instead of using only one dense feed-forward network for every token in every relevant layer, an MoE layer contains multiple expert networks and a router that selects which expert or experts process a token.
Input token representation
           ↓
      Router / gating
           ↓
   Select expert network(s)
           ↓
    Expert computation
           ↓
     Output representation


This can increase the model's total parameter capacity without requiring every expert's weights to be used for every token. The trade-offs include routing complexity, communication overhead, and expert load balancing.

For Llama 4 specifically, the published Hugging Face documentation describes Scout and Maverick as MoE models with 17 billion active parameters per token, while their total parameter counts differ substantially. 

Hugging Face




Interview trap: Does a model with 400 billion total parameters perform the full computation of all 400 billion parameters for every token?

Not necessarily. In an MoE model, only selected experts are activated for a given token, although shared components and routing operations also contribute to the computation.
7. LLaMA vs GPT vs BERT
Feature	LLaMA	GPT-style decoder LLM	BERT
Architecture	Decoder-only Transformer	Typically decoder-only Transformer	Encoder-only Transformer
Attention	Causal self-attention in classic text models	Causal self-attention	Usually bidirectional self-attention
Common objective	Next-token prediction	Next-token prediction	Masked-language modeling in original BERT
Text generation	Yes	Yes	Not designed for standard autoregressive generation
Typical applications	Chat, RAG, summarization, coding	Chat, RAG, summarization, coding	Classification, embeddings, token labeling
Weights available for local use	Depends on model and license	Depends on the model	Depends on model and license
A key distinction is that LLaMA is a model family, whereas GPT and BERT refer to different model families or naming conventions. Not every GPT-style model has the same architecture, and model availability or licensing depends on the specific release.
8. How LLaMA is used in a real GenAI application
Since LLaMA is a language model, it is important to distinguish the model itself from the complete GenAI application built around it.

Consider an enterprise document question-answering system using RAG.
User asks a question

"What is our company's leave policy?"

Embedding model + vector database

Retrieve relevant authorized document chunks

Optional reranking

Improve relevance of retrieved chunks

LLaMA

Reads the question and retrieved context, then generates an answer

Application response

Answer with relevant citations or document references


In this architecture:
- The embedding model converts text into vectors for retrieval.
- The vector database stores and searches document representations.
- The retriever selects relevant chunks.
- The LLaMA model generates the final response using the supplied context.
- Application-level security enforces access permissions and protects confidential data.
Important interview point: LLaMA does not inherently need a vector database. RAG is an application pattern that can use LLaMA or another compatible LLM. Similarly, a vector database does not itself generate answers.
Local inference using Ollama
You can experiment with a Llama model locally using Ollama. After choosing and downloading a supported model, a typical command-line interaction looks like this:
ollama run llama3.2


Then enter a prompt, for example:
Explain the difference between RAG and fine-tuning.


The exact model tag must match a model available in your Ollama installation. Local execution depends on available RAM, model size, quantization, and hardware.

This is useful for learning inference, prompting, and RAG without starting with a paid hosted LLM API.
9. Advanced interview questions and answers
These are the questions I would prioritize for a GenAI engineering interview.
Interview practice
0 of 18 answers revealed

Try answering each question aloud before revealing the answer.
1Why is LLaMA called an autoregressive model?Reveal answer

2Why does a decoder-only model need a causal mask?Reveal answer

3What is the difference between an embedding and a hidden state?Reveal answer

4What is the role of Q, K and V?Reveal answer

5Why is attention divided by the square root of the key dimension?Reveal answer

6What is the difference between RoPE and learned absolute positional embeddings?Reveal answer

7Why does LLaMA use RMSNorm?Reveal answer

8What is the purpose of SwiGLU?Reveal answer

9What is GQA, and why is it useful?Reveal answer

10What is a KV cache?Reveal answer

11Does a higher parameter count guarantee better answers?Reveal answer

12Does increasing context length always improve RAG?Reveal answer

13What is the difference between pretraining and fine-tuning?Reveal answer

14Can fine-tuning eliminate hallucinations?Reveal answer

15How does an MoE model differ from a dense Transformer?Reveal answer

16How would you reduce LLaMA inference memory?Reveal answer

17What is the difference between quantization and LoRA?Reveal answer

18Can LLaMA answer questions about private documents without RAG or fine-tuning?Reveal answer
Reveal all answers
10. Common mistakes to avoid in interviews
Incorrect statement	Correct understanding
LLaMA is an encoder-decoder model.	Classic text LLaMA is decoder-only.
Token IDs are embeddings.	IDs are integers; embeddings are vectors.
Attention always sees the entire sequence.	Causal attention blocks future positions.
RoPE and token embeddings are the same thing.	Token embeddings represent tokens; RoPE injects positional information into attention.
Softmax produces the final answer.	Softmax gives a probability distribution; decoding selects a token, and generation continues.
RAG retrains the model for every question.	RAG retrieves context at inference time; it does not inherently retrain the LLM.
Quantization reduces the number of model parameters.	It typically reduces the precision used to represent weights, not the number of parameter entries.
All LLaMA generations use identical architecture.	Architectural features and configurations vary by generation and checkpoint.
11. Final revision sheet
Use this as your quick revision list before the interview.
Architecture

Decoder-only Transformer for classic text LLaMA


Input processing

Tokenizer → token IDs → embeddings


Attention

Q, K, V → scaled dot-product attention → causal masking


Position information

RoPE


Normalization

RMSNorm


Feed-forward network

SwiGLU in classic LLaMA


Deep-network stability

Residual connections


Output

Final normalization → linear projection → vocabulary logits


Generation

Autoregressive next-token prediction


Inference optimization

KV cache, GQA, quantization


Training

Pretraining → instruction fine-tuning → optional preference alignment


Advanced generation

Llama 4 includes MoE and multimodal capabilities



Your 60-second interview answer
"LLaMA is a family of language models developed by Meta. Classic text LLaMA uses a decoder-only Transformer architecture and generates text through autoregressive next-token prediction. The input passes through tokenization and embedding layers, followed by repeated Transformer blocks. Each block uses causal self-attention to learn relationships between preceding tokens, RoPE to provide positional information, RMSNorm for normalization, residual connections, and a SwiGLU feed-forward network. The final linear layer produces vocabulary logits, and a decoding strategy selects the next token. For efficient inference, implementations can use GQA, KV caching, and quantization. The model can be instruction-tuned for conversational tasks and integrated with a retriever and vector database to build RAG applications. Newer generations, such as Llama 4, also introduce MoE and multimodal architectural features."