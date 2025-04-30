# Flexible Inference Mechanisms for SMILES Generation

This document describes how to use the inference mechanisms implemented for generating SMILES from spectral data.

## Overview

The implementation supports five different decoding strategies:

1. **Greedy Decoding**: Selects the most probable token at each step. Simple but may result in suboptimal sequences.
2. **Beam Search**: Maintains multiple hypotheses and selects the most probable sequence. Usually produces higher quality results than greedy decoding.
3. **Sampling**: Samples from the distribution of tokens, controlled by a temperature parameter.
4. **Nucleus Sampling (top-p)**: Samples from the smallest set of tokens whose cumulative probability exceeds a threshold p.
5. **Entropix**: Adaptive decision-making based on entropy and variance of token distributions, with dynamic layer looping for uncertain predictions.

## Understanding Beam Search

Beam search is an extension of greedy search that keeps track of multiple promising sequences (the "beam") during generation. Instead of only selecting the most probable token at each step, beam search:

1. Starts with a single sequence (the BOS token)
2. Generates multiple possible continuations (beam_width)
3. Keeps only the most promising beam_width sequences
4. Repeats until all sequences reach EOS or max length

Advantages:
- More thorough exploration of the search space
- Often produces higher quality outputs than greedy decoding
- Can return multiple diverse candidates

Disadvantages:
- Significantly slower than greedy decoding
- Higher memory usage
- Not guaranteed to find the globally optimal sequence

### Length Penalty in Beam Search

Beam search tends to favor shorter sequences, which can be problematic for chemical structure generation. To address this, a length penalty parameter is included:

- `length_penalty` < 1.0: Favors shorter sequences
- `length_penalty` = 1.0: Neutral (no penalty)
- `length_penalty` > 1.0: Favors longer sequences

## Understanding Sampling-based Decoding

While greedy and beam search are deterministic, sampling-based methods introduce randomness:

1. **Temperature Sampling**: The "temperature" parameter controls randomness:
   - Lower values (e.g., 0.7) make sampling more conservative
   - Higher values (e.g., 1.5) make sampling more diverse and random

2. **Nucleus Sampling (top-p)**: Instead of sampling from all tokens, nucleus sampling:
   - Sorts tokens by probability
   - Keeps only the smallest set of tokens whose cumulative probability exceeds p
   - Samples from this reduced set
   - This dynamically adapts the vocabulary size based on confidence

## Understanding Entropix Decoding

Entropix is an adaptive decoding strategy that uses entropy-based metrics to make dynamic decisions during sequence generation:

1. **Entropy and Varentropy**: 
   - **Entropy**: Measures the uncertainty in the token distribution
   - **Varentropy**: Measures the variance of log probabilities in the distribution

2. **Decision Logic**:
   - Low entropy, low varentropy: High confidence, explore multiple candidates (branch)
   - Low entropy, high varentropy: Clear best choice, take the most probable token
   - High entropy: Uncertain prediction, apply additional processing (layer looping)

3. **Layer Looping**: For uncertain predictions, the decoder layers are executed multiple times to refine the prediction:
   - Increases computation time but improves quality in uncertain regions
   - The maximum number of loops is controlled by the `max_loops` parameter
   - Loops are dynamically increased when consecutive tokens have high entropy

### When to Use Entropix

Entropix is particularly useful for chemical structure generation, where:
- Some parts of the SMILES string may be very predictable (e.g., common functional groups)
- Other parts may require more careful consideration (e.g., complex stereochemistry)
- Hard decisions need to be made with incomplete information

The algorithm adapts its behavior based on the confidence in each prediction, balancing quality and efficiency.

## Calibrating Entropix Thresholds

To optimize Entropix for your specific model and dataset, you can use the calibration tool:

```bash
python calibrate_entropix.py --checkpoint <path_to_checkpoint> \
                             --config <path_to_config> \
                             --split test \
                             --num_samples 100 \
                             --output_dir calibration_results
```

This tool will:
1. Analyze entropy and varentropy distributions across your test dataset
2. Recommend threshold values based on statistical analysis
3. Generate visualizations of the distributions
4. Save detailed results for further analysis

Based on the calibration, you'll get recommended thresholds:
- Conservative: 50th percentile values (balanced decision-making)
- Moderate: 75th percentile values (more aggressive branching/looping)
- Aggressive: 90th percentile values (maximum quality, slower generation)

## When to Use Each Strategy

| Strategy | Best Used When | Pros | Cons |
|----------|----------------|------|------|
| Greedy | You need fast results; the task has clear right answers | Fastest; simple | Often suboptimal results |
| Beam Search | You need high-quality, accurate sequences | Better quality than greedy; multiple candidates | Slower; higher memory usage |
| Temperature Sampling | You want some diversity; exploring possible results | Introduces controlled randomness | Results vary between runs |
| Nucleus Sampling | You want creative but still high-quality outputs | Better than pure sampling for maintaining quality while being diverse | Results vary between runs; parameter tuning needed |
| Entropix | You need accurate chemical structures with complex patterns | Adapts to uncertainty; selective computation | Requires calibration; more complex implementation |

## Usage

### From Command Line

You can use the search_and_infer.py script with the decoding options:

```bash
python search_and_infer.py --checkpoint <path_to_checkpoint> \
                          --config <path_to_config> \
                          --strategy <greedy|beam|sampling|nucleus|entropix> \
                          --beam_width 5 \
                          --temperature 1.0 \
                          --top_k 0 \
                          --top_p 0.0 \
                          --length_penalty 1.0 \
                          --entropy_threshold 1.0 \
                          --varentropy_threshold 0.5 \
                          --max_loops 3 \
                          --raw_nmr <path_to_nmr_file> \
                          --raw_ir <path_to_ir_file>
```

### Testing Different Strategies

You can use the test_inference.py script to compare different decoding strategies:

```bash
python test_inference.py --checkpoint <path_to_checkpoint> \
                        --config <path_to_config> \
                        --raw_nmr <path_to_nmr_file> \
                        --raw_ir <path_to_ir_file>
```

Or, to test with data from the dataset:

```bash
python test_inference.py --checkpoint <path_to_checkpoint> \
                        --dataset_test \
                        --split test \
                        --index 0
```

This will run all four decoding strategies and display their results.

### Simple Example

For a more straightforward demonstration, use the example_inference.py script:

```bash
python example_inference.py --checkpoint <path_to_checkpoint> \
                           --raw_nmr <path_to_nmr_file> \
                           --raw_ir <path_to_ir_file>
```

### In Your Own Code

To use the inference mechanisms in your code:

```python
from inference import ModelInference, DecodingStrategy
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer

# Initialize model and tokenizer
model = MultiModalToSMILESModel(...)
tokenizer = SmilesTokenizer(...)

# Create inference wrapper
inference = ModelInference(model, tokenizer, device)

# Run inference with different strategies
results = inference.decode(
    nmr_tokens=nmr_tokens,
    ir_data=ir_data,
    strategy=DecodingStrategy.BEAM,  # Or GREEDY, SAMPLING, NUCLEUS, ENTROPIX
    max_len=128,
    beam_width=5,  # Only for beam search
    temperature=1.0,  # For sampling and nucleus
    top_k=0,  # For sampling
    top_p=0.9,  # For nucleus sampling
    length_penalty=1.0,  # For beam search
    entropy_threshold=1.0,  # For entropix
    varentropy_threshold=0.5,  # For entropix
    max_loops=3  # For entropix
)
```

## Parameters

- **strategy**: Decoding strategy to use (greedy, beam, sampling, nucleus, entropix)
- **max_len**: Maximum sequence length
- **beam_width**: Number of beams to maintain in beam search
- **temperature**: Temperature for sampling (higher = more random)
- **top_k**: If > 0, only sample from the top k tokens
- **top_p**: If > 0, sample from the smallest set of tokens whose cumulative probability exceeds p
- **length_penalty**: Penalty factor for sequence length in beam search
- **repetition_penalty**: Penalty for repeating tokens (not fully implemented yet)
- **entropy_threshold**: Threshold for entropy in Entropix (higher values mean fewer branches)
- **varentropy_threshold**: Threshold for varentropy in Entropix (higher values mean more greedy decisions)
- **max_loops**: Maximum number of decoder layer loops for Entropix

## Recommended Parameter Settings

Based on experiments and common practices:

### For Beam Search
- Start with `beam_width=5` (higher values like 10-20 can be better but slower)
- Use `length_penalty=1.0` initially, adjust if sequences are too short

### For Sampling
- For more conservative sampling: `temperature=0.7`
- For more diverse sampling: `temperature=1.2`

### For Nucleus Sampling
- Good starting point: `top_p=0.9, temperature=1.0`
- More conservative: `top_p=0.7, temperature=0.8`
- More diverse: `top_p=0.95, temperature=1.2`

### For Entropix
- After calibration, use the recommended thresholds:
  - Conservative: Use 50th percentile values (typically around `entropy_threshold=1.0, varentropy_threshold=0.5`)
  - Moderate: Use 75th percentile values
  - Aggressive: Use 90th percentile values
- Start with `max_loops=3` for a good balance of quality and speed

## Performance Considerations

- Beam search is significantly slower than greedy decoding:
  - beam_width=5 is ~5x slower than greedy
  - beam_width=10 is ~10x slower than greedy
- Memory usage increases linearly with beam width
- For very long sequences, consider using greedy or sampling
- On CPU, all methods will be significantly slower - prefer GPU if available
- Entropix performance varies based on input complexity:
  - May approach greedy speed for simple inputs
  - Can be slower than beam search for highly uncertain inputs due to layer looping
  - Overall, provides a good balance of quality and speed

## Reproducibility

For sampling-based methods, you can set a seed to make results reproducible:

```python
import torch
torch.manual_seed(42)  # Set seed before running inference
```

## Implementation Details

The beam search algorithm in this implementation:
1. Tracks sequence probabilities using log probabilities to avoid underflow
2. Uses a penalty based on sequence length to avoid bias toward shorter sequences
3. Returns the beam_width most likely sequences, ordered by score 

The Entropix algorithm in this implementation:
1. Uses entropy and varentropy to make branching decisions
2. Dynamically increases decoder layer loops for uncertain predictions
3. Balances exploration (branching) and exploitation (greedy) based on calibrated thresholds
4. Prioritizes computation where it's most needed, saving resources on more certain predictions 