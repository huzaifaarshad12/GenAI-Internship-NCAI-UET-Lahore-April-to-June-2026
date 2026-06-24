# Day 6 – RNNs, LSTMs, and GRUs

## 1. Recurrent Neural Networks (RNN)
**Concept:** RNNs are designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or spoken words. Unlike standard feedforward neural networks, RNNs have a "memory" which captures information about what has been calculated so far.
*   **Hidden State:** The network passes a hidden state from one time step to the next, representing the network's memory.

```mermaid
graph LR
    x_t-1 --> RNN_t-1
    RNN_t-1 --> h_t-1
    RNN_t-1 --> RNN_t
    x_t --> RNN_t
    RNN_t --> h_t
    RNN_t --> RNN_t+1
    x_t+1 --> RNN_t+1
    RNN_t+1 --> h_t+1
```

## 2. Vanishing Gradient Problem & Temporal Dependencies
*   **Vanishing Gradient:** When training artificial neural networks with gradient-based learning methods and backpropagation, the gradients (useful for weight updates) often get smaller and smaller as they propagate backward through time. This stops the early layers from learning.
*   **Temporal Dependencies:** Because of the vanishing gradient, vanilla RNNs struggle to learn long-range temporal dependencies (e.g., remembering a word from the beginning of a long paragraph).

## 3. Long Short-Term Memory (LSTM)
**Concept:** LSTMs are a special kind of RNN, explicitly designed to avoid the long-term dependency problem. They introduce a cell state ($C_t$) and three gates to carefully regulate the flow of information.
*   **Forget Gate:** Decides what information we're going to throw away from the cell state.
*   **Input Gate:** Decides what new information we're going to store in the cell state.
*   **Output Gate:** Decides what we're going to output based on our filtered cell state.

```mermaid
flowchart TD
    subgraph LSTM Cell
    F[Forget Gate]
    I[Input Gate]
    O[Output Gate]
    C_update[Cell State Update]
    end
    
    h_prev[Previous Hidden State, h_t-1] --> F & I & O
    x_t[Input, x_t] --> F & I & O

    C_prev[Previous Cell State, C_t-1] --> F
    F --> C_t[New Cell State, C_t]
    I --> C_update
    C_update --> C_t
    
    C_t --> O
    O --> h_t[New Hidden State, h_t]
```

## 4. Gated Recurrent Unit (GRU)
**Concept:** The GRU is a newer generation of RNNs and is comparable to LSTM. They effectively solve the vanishing gradient problem using a simpler architecture (fewer parameters).
*   **Update Gate:** Similar to the forget and input gates of an LSTM model combined. It decides what information to throw away and what new information to add.
*   **Reset Gate:** Decides how much of the past information to forget.

```mermaid
flowchart TD
    subgraph GRU Cell
    R[Reset Gate]
    U[Update Gate]
    H_cand[Candidate Hidden State]
    end
    
    h_prev[Previous Hidden State, h_t-1] --> R & U & H_cand
    x_t[Input, x_t] --> R & U & H_cand
    
    R -.-> H_cand
    U --> h_t[New Hidden State, h_t]
    H_cand --> h_t
```
