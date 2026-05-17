# part-3-nlp-sequence-modeling
### Task 3: Text Vectorization
**Why text must be converted into vectors before being used by a model**  
Machine learning and Deep learning models work on matrix multiplications, dot products, additions, and calculus-based algorithms (like gradient descent). For these models to process languages, it maps text characters/words into a standardized numerical format. To capture the meaning and structure, advanced vectorization (like tokenizer-based sequences and word embeddings) assigns distinct numbers or coordinate locations to words. This preserves their order, contextual sequence position, and structural relationships, allowing deep learning models to mathematically detect combinations of words.

### Task 5: Sequence Model or Conceptual Architecture
This model (LSTM) processes text sequentially, allowing the network to understand context, word order, and long-range sentences. 
- The input sequence layer converts every customer message into an array of 20 integer tokens during pre-processing pipeline.
- The embedding layer transforms the integer word tokens into dense, continuous vector spaces. The mathematical distances represent semantic meanings.
- LSTM layer takes in the 20 word vectors one by one, and at each step updates the cell state and hidden states.
- Dropout layer prevents the network from over-memorizing phrasing patterns, and forces it to learn sentiment markers and preventing overfitting.
- The output layer maps the summarized context from the LSTM model into the 3 sentiment classes (negative, neutral, positive).
- The loss function (loss='sparse_categorical_crossentropy') measures the error between the model's predicted probability distribution and the true target labels.
- The optimizer uses Adam (Adaptive Moment Estimation) to calculate individual learning rates for every parameter in the neural network during training.

### Task 6: Attention and Transformer Reflection
**Why RNNs struggle with long-term dependencies**  
An RNN processes text one word at a time while carrying a summary (hidden state) of the previous words forward. Every time it reads the next new word, the RNN multiplies it with the old summary. Mathematically, during training, the network multiplies numbers over and over again to pass information backward (back propagation). Repeatedly multiplying these gradients and weights, the values almost reaches zero. This is the vanishing gradient problem. By the time the RNN reaches the end of a long paragraph, the influence of the first few words is completely vanished.

**How LSTMs help with memory**  
Long short term memory networks introduce dedicated cell states controlled by forget, input and output gates. These gates use sigmoid activations to determine what information to forget or keep, thus preventing gradients from vanishing.

**What attention solves in sequence-to-sequence tasks**  
Sequence-to-Sequence tasks usually involve taking an input sequence (like an English sentence) and turning it into an output sequence (like a French translation). Traditional encoder-decoder systems tried to compress a big sentence into a single, fixed size vector. If the sentence was too long, details were lost. Attention solves this issue by allowing the model to look back across all original English words (historical encoder hidden states) and decide which words are most relevant.

**Why transformers are important in modern NLP and Generative AI**  
Even though LSTMs and Attention solved memory problems, they were slow. RNNs and LSTMs process text sequentially (word-by-word), which means they cannot move on to the next word until they are completely finished calculating the previous word. A transformer uses a mechanism called self attention. Words connect with other words in the rest of the sentence, to form context. Because it calculates these relationships for all words all at once, it builds a complete map of context instantly rather than trying to remember it at every step. This allows models to train on a massive scale and at unprecedented speeds.

Dataset Source: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
