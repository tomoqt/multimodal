import torch
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any
from enum import Enum
import heapq


class DecodingStrategy(Enum):
    """Enum for different decoding strategies."""
    GREEDY = "greedy"
    GREEDY_LOOP = "greedy_loop"
    BEAM = "beam"
    SAMPLING = "sampling"
    NUCLEUS = "nucleus"
    ENTROPIX = "entropix"


class BeamSearchNode:
    """
    Node in the beam search process.
    Each node represents a partial sequence with an associated score.
    The score is based on log probability and length normalization.
    """
    def __init__(self, hidden_state, prev_node, token_id, log_prob, length):
        """
        Initialize a beam search node.
        
        Args:
            hidden_state: The hidden state (not used in current implementation)
            prev_node: The previous node in the sequence
            token_id: The token ID for this node
            log_prob: The cumulative log probability of the sequence so far
            length: The length of the sequence so far
        """
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
    
    def eval(self, alpha=1.0):
        """
        Evaluate the node's score with length normalization.
        Uses the Google NMT length normalization formula:
        score = log_prob / ((5 + length)^alpha / (5 + 1)^alpha)
        
        Args:
            alpha: Length penalty factor. Values > 1.0 favor longer sequences,
                  values < 1.0 favor shorter sequences, 1.0 means no penalty.
            
        Returns:
            The score with length normalization applied
        """
        # Apply length normalization using Google's GNMT formula
        if alpha == 1.0:
            return self.log_prob  # No length normalization
        
        # Length penalty: ((5 + length)^alpha / (5 + 1)^alpha)
        # Constants 5 and 1 are from the original GNMT paper
        length_penalty = ((5 + self.length) ** alpha) / ((5 + 1) ** alpha)
        return self.log_prob / length_penalty
    
    def __lt__(self, other):
        """
        Compare nodes for sorting in priority queue.
        Lower score means lower priority in the beam.
        
        Args:
            other: Another BeamSearchNode to compare with
            
        Returns:
            True if this node has lower score than other
        """
        return self.eval() < other.eval()
    
    def __gt__(self, other):
        """
        Compare nodes for sorting in reverse order.
        Higher score means higher priority in the beam.
        
        Args:
            other: Another BeamSearchNode to compare with
            
        Returns:
            True if this node has higher score than other
        """
        return self.eval() > other.eval()


class EntropixNode:
    """
    Node in the Entropix tree search process.
    Each node represents a partial sequence with associated log probabilities
    and metrics for decision-making.
    """
    def __init__(self, hidden_state, prev_node, token_id, log_prob, length, curr_loops=1):
        """
        Initialize an Entropix search node.
        
        Args:
            hidden_state: The hidden state (not used in current implementation)
            prev_node: The previous node in the sequence
            token_id: The token ID for this node
            log_prob: The cumulative log probability of the sequence so far
            length: The length of the sequence so far
            curr_loops: Current number of loops used for this node
        """
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
        self.curr_loops = curr_loops  # Track how many loops were used
    
    def eval(self):
        """
        Evaluate the node's score.
        Simple evaluation based on log probability.
        
        Returns:
            The score based on log probability
        """
        return self.log_prob
    
    def __lt__(self, other):
        """
        Compare nodes for sorting in priority queue.
        Lower score means lower priority in the queue.
        
        Args:
            other: Another EntropixNode to compare with
            
        Returns:
            True if this node has lower score than other
        """
        return self.eval() < other.eval()
    
    def __gt__(self, other):
        """
        Compare nodes for sorting in reverse order.
        Higher score means higher priority in the queue.
        
        Args:
            other: Another EntropixNode to compare with
            
        Returns:
            True if this node has higher score than other
        """
        return self.eval() > other.eval()


class ModelInference:
    """
    A wrapper class for model inference that supports different decoding strategies.
    """
    
    def __init__(self, model, tokenizer, device=None, ir_as_prompt=False):
        """
        Initialize the inference wrapper.
        
        Args:
            model: The model to use for inference
            tokenizer: The tokenizer for decoding
            device: The device to run inference on
            ir_as_prompt: Whether IR data is used as prompt tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.ir_as_prompt = ir_as_prompt
        
        # Cache token IDs for efficiency
        self.bos_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
    
    def decode(
        self,
        nmr_tokens: Optional[torch.Tensor] = None,
        ir_data: Optional[torch.Tensor] = None,
        mass_data: Optional[torch.Tensor] = None,
        strategy: DecodingStrategy = DecodingStrategy.GREEDY,
        max_len: int = 128,
        beam_width: int = 5,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        entropy_threshold: float = 1.0,
        varentropy_threshold: float = 0.5,
        max_loops: int = 3,
        automatic_loop_exit: bool = False,
        automatic_loop_exit_threshold: float = 0.01,
        loop_increase_step: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Decode using the specified strategy.
        
        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data or tokenized IR data (if ir_as_prompt=True)
            mass_data: Mass spectrometry data
            strategy: The decoding strategy to use
            max_len: Maximum sequence length
            beam_width: Beam width for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            length_penalty: Length penalty for beam search
            repetition_penalty: Penalty for repeating tokens
            entropy_threshold: Threshold for entropy in Entropix decoding
            varentropy_threshold: Threshold for variance of logprobs in Entropix decoding
            max_loops: Maximum number of times to loop the middle layer
            automatic_loop_exit: Whether to automatically exit loops based on entropy
            automatic_loop_exit_threshold: Threshold for automatic loop exit
            loop_increase_step: Amount to increase loop count by when in high-entropy (non-automatic mode) in Entropix
            
        Returns:
            List of decoded sequences or Tuple of (List of decoded sequences, List of loop representations)
        """
        if strategy == DecodingStrategy.GREEDY:
            return self.greedy_decode(nmr_tokens, ir_data, mass_data, max_len)
        elif strategy == DecodingStrategy.GREEDY_LOOP:
            return self.greedy_decode_with_loops(
                nmr_tokens, 
                ir_data, 
                mass_data, 
                max_len, 
                num_loops=kwargs.get("num_loops", 1),
                loops_representation=kwargs.get("loops_representation", False)
            )
        elif strategy == DecodingStrategy.BEAM:
            return self.beam_search(nmr_tokens, ir_data, mass_data, max_len, beam_width, length_penalty)
        elif strategy in [DecodingStrategy.SAMPLING, DecodingStrategy.NUCLEUS]:
            return self.sample_decode(nmr_tokens, ir_data, mass_data, max_len, temperature, top_k, top_p)
        elif strategy == DecodingStrategy.ENTROPIX:
            return self.entropix_decode(nmr_tokens, ir_data, mass_data, max_len, top_k,
                                       entropy_threshold, varentropy_threshold, max_loops, automatic_loop_exit, automatic_loop_exit_threshold, loop_increase_step)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")
    
    def prepare_inputs(self, nmr_tokens, ir_data, mass_data):
        """
        Prepare inputs for the model by adding batch dimension if needed.
        
        Returns:
            Tuple of prepared inputs and batch size
        """
        # Prepare batch dimension
        if nmr_tokens is not None and nmr_tokens.dim() == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if ir_data is not None and ir_data.dim() == 1:
            ir_data = ir_data.unsqueeze(0)
        if mass_data is not None and mass_data.dim() == 1:
            mass_data = mass_data.unsqueeze(0)
            
        # Infer batch size from the first non-None input
        batch_size = None
        for tensor in [nmr_tokens, ir_data, mass_data]:
            if tensor is not None:
                batch_size = tensor.shape[0]
                break
                
        if batch_size is None:
            batch_size = 1
            
        return (nmr_tokens, ir_data, mass_data), batch_size
    
    def encode_inputs(self, nmr_tokens, ir_data, mass_data):
        """
        Encode the spectral data using the model's encoder.
        
        Returns:
            Encoded memory from the model's encoder
        """
        # Determine batch size from inputs
        batch_size = 1
        if nmr_tokens is not None:
            batch_size = nmr_tokens.shape[0]
        elif ir_data is not None:
            batch_size = ir_data.shape[0]
        elif mass_data is not None:
            batch_size = mass_data.shape[0]
        
        # Handle memory creation based on ir_as_prompt mode
        if self.ir_as_prompt:
            # In ir_as_prompt mode, we still need to return a properly sized memory tensor
            # for use in the transformer decoder, even if it's just zeros
            memory = torch.zeros(
                batch_size, 
                self.model.decoder.max_memory_length, 
                self.model.decoder.memory_dim, 
                device=self.device
            )
            return memory
        else:
            # Standard encoder-based approach for raw IR data values
            # The model's encoder will process the spectral data
            memory = self.model.encoder(mass_data, ir_data, None)
            
            if memory is None:
                memory = torch.zeros(
                    batch_size, 
                    self.model.decoder.max_memory_length, 
                    self.model.decoder.memory_dim, 
                    device=self.device
                )
            
            return memory
    
    def postprocess_sequences(self, sequences):
        """
        Postprocess generated sequences by removing BOS token and decoding.
        
        Args:
            sequences: List of token ID sequences
            
        Returns:
            List of decoded strings
        """
        decoded_sequences = []
        for seq in sequences:
            try:
                # Find EOS token if present
                eos_idx = seq.index(self.eos_token_id)
                seq = seq[:eos_idx]
            except ValueError:
                # No EOS token found
                pass
            # Skip BOS token during decoding
            decoded = self.tokenizer.decode(seq[1:])
            decoded_sequences.append(decoded)
        return decoded_sequences
    
    def greedy_decode(self, nmr_tokens, ir_data, mass_data=None, max_len=128):
        """
        Greedy decoding strategy - selects the most probable token at each step.
        
        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            
        Returns:
            List of decoded sequences
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare inputs (add batch dimension if needed)
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            
            # Encode spectral data
            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Initialize with start token
            current_token = torch.tensor([[self.bos_token_id]] * batch_size, device=self.device)
            
            # For each batch, keep track of generated tokens
            generated_sequences = [[self.bos_token_id] for _ in range(batch_size)]
            
            # For each sequence position
            for _ in range(max_len):
                # Get logits from the model
                logits = self.model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens)
                
                # Select the most probable token
                next_token = logits[:, -1:].argmax(dim=-1)
                
                # Add to generated sequences
                for i in range(batch_size):
                    token_id = next_token[i].item()
                    generated_sequences[i].append(token_id)
                    # Early stopping if EOS token is generated
                    if token_id == self.eos_token_id:
                        break
                
                # Prepare for next iteration
                current_token = torch.cat((current_token, next_token), dim=1)
            
            # Post-process and decode the sequences
            return self.postprocess_sequences(generated_sequences)
    
    def beam_search(self, nmr_tokens, ir_data, mass_data=None, max_len=128, beam_width=5, length_penalty=1.0):
        """
        Beam search decoding - maintains top-k candidate sequences at each step.
        
        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            beam_width: Number of candidate sequences to maintain
            length_penalty: Penalty factor for sequence length
            
        Returns:
            List of decoded sequences
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare inputs (for simplicity, only handle batch size 1)
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            if batch_size != 1:
                raise ValueError("Beam search currently only supports batch size 1")
            
            # Encode spectral data
            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Start with BOS token
            node = BeamSearchNode(None, None, self.bos_token_id, 0, 0)
            current_token = torch.tensor([[self.bos_token_id]], device=self.device)
            
            # Initial step - get logits for the first token
            logits = self.model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens)
            logits = logits[0, -1, :]  # (vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get top beam_width candidates
            topk_log_probs, topk_indices = log_probs.topk(beam_width)
            
            # Initialize beam
            nodes = []
            for i in range(beam_width):
                token_id = topk_indices[i].item()
                log_prob = topk_log_probs[i].item()
                nodes.append(BeamSearchNode(None, node, token_id, log_prob, 1))
            
            # Keep track of sequences that have reached EOS
            endnodes = []
            
            # Beam search
            for step in range(2, max_len + 1):
                # Check if we can terminate
                if len(endnodes) >= beam_width:
                    break
                
                # Get list of all candidate sequences for this step
                candidates = []
                
                # For each node in the beam
                for node in nodes:
                    # If this node contains EOS token, add to endnodes
                    if node.token_id == self.eos_token_id:
                        endnodes.append(node)
                        continue
                    
                    # Create sequence for this node and get next token distribution
                    seq = []
                    n = node
                    while n.prev_node:
                        seq.append(n.token_id)
                        n = n.prev_node
                    seq.append(self.bos_token_id)
                    seq.reverse()
                    
                    # Convert to tensor and get model predictions
                    seq_tensor = torch.tensor([seq], device=self.device)
                    
                    # Get the next token distribution
                    logits = self.model.decoder(tgt=seq_tensor, memory=memory, nmr_tokens=nmr_tokens)
                    logits = logits[0, -1, :]  # (vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Get top beam_width candidates and add to candidate list
                    topk_log_probs, topk_indices = log_probs.topk(beam_width)
                    for i in range(beam_width):
                        token_id = topk_indices[i].item()
                        log_prob = topk_log_probs[i].item()
                        candidates.append(BeamSearchNode(
                            hidden_state=None,
                            prev_node=node,
                            token_id=token_id,
                            log_prob=node.log_prob + log_prob,
                            length=node.length + 1
                        ))
                
                # Reached EOS but don't have enough endnodes
                if len(endnodes) >= beam_width:
                    break
                
                # Sort candidates by score and keep top beam_width
                candidates.sort(reverse=True)
                nodes = candidates[:beam_width]
            
            # If no complete sequences, take the best from current beam
            if len(endnodes) == 0:
                endnodes = nodes
            
            # Sort endnodes by score and prepare return format
            endnodes.sort(reverse=True)
            
            # Reconstruct the sequences
            sequences = []
            for node in endnodes[:beam_width]:
                seq = []
                n = node
                while n.prev_node:
                    seq.append(n.token_id)
                    n = n.prev_node
                seq.append(self.bos_token_id)
                seq.reverse()
                sequences.append(seq)
            
            # Post-process and decode the sequences
            return self.postprocess_sequences(sequences)
    
    def sample_decode(self, nmr_tokens, ir_data, mass_data=None, max_len=128, temperature=1.0, top_k=0, top_p=0.0):
        """
        Sample-based decoding - samples tokens based on their probability distribution.
        
        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            temperature: Temperature for sampling (higher = more random)
            top_k: If > 0, only sample from the top k tokens
            top_p: If > 0, sample from the smallest set of tokens whose cumulative probability exceeds p
            
        Returns:
            List of decoded sequences
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare inputs
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            
            # Start token for decoding
            current_token = torch.tensor([[self.bos_token_id]] * batch_size, device=self.device)
            
            # Encode spectral data
            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Initialize generation
            generated_sequences = [[self.bos_token_id] for _ in range(batch_size)]
            max_len = min(max_len, self.model.decoder.max_seq_length)
            finished_sequences = [False] * batch_size
            
            # Generate tokens step by step
            for _ in range(max_len):
                logits = self.model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens)
                logits = logits[:, -1, :] / max(temperature, 1e-6)  # Apply temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    # Zero out all logits below the top k ones
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Apply boolean mask to sorted indices to keep track of which indices to remove
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Update sequences
                for i in range(batch_size):
                    if not finished_sequences[i]:
                        token = next_token[i].item()
                        generated_sequences[i].append(token)
                        if token == self.eos_token_id:
                            finished_sequences[i] = True
                
                # Stop if all sequences have EOS
                if all(finished_sequences):
                    break
                    
                # Update input for next iteration
                current_token = torch.cat([current_token, next_token], dim=1)
            
            return self.postprocess_sequences(generated_sequences)
    
    def calculate_entropy(self, logprobs):
        """
        Calculate the entropy of a probability distribution from log probabilities.
        
        Args:
            logprobs: Tensor of log probabilities
            
        Returns:
            Entropy value as a float
        """
        # Convert logprobs to probabilities
        probs = torch.exp(logprobs)
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * logprobs)
        return entropy.item()
    
    def calculate_varentropy(self, logprobs):
        """
        Calculate the varentropy (variance of log probabilities) of a distribution.
        
        Args:
            logprobs: Tensor of log probabilities
            
        Returns:
            Varentropy value as a float
        """
        # Convert logprobs to probabilities
        probs = torch.exp(logprobs)
        # Calculate mean logprob weighted by probabilities
        mean_logprob = torch.sum(probs * logprobs)
        # Calculate variance: sum(p * (logp - mean_logp)^2)
        varentropy = torch.sum(probs * (logprobs - mean_logprob) ** 2)
        return varentropy.item()
    
    def entropix_decode(self, nmr_tokens, ir_data, mass_data=None, max_len=128, top_k=5, 
                       entropy_threshold=1.0, varentropy_threshold=0.5, max_loops=3, 
                       automatic_loop_exit=False, automatic_loop_exit_threshold=0.01, 
                       loop_increase_step=None):
        """
        Entropix tree search - uses entropy and varentropy to make branching decisions.
        
        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            top_k: Number of top candidates to consider for branching
            entropy_threshold: Threshold for entropy (low/high decision boundary)
            varentropy_threshold: Threshold for varentropy (low/high decision boundary)
            max_loops: Maximum number of times to loop the middle layer
            automatic_loop_exit: Whether to automatically exit loops based on entropy
            automatic_loop_exit_threshold: Threshold for automatic loop exit
            loop_increase_step: Amount to increase loop count by when in high-entropy (non-automatic mode)
            
        Returns:
            Tuple: (List of decoded sequences, List of loop counts per sequence)
            where loop counts per sequence is a list of lists, each inner list containing
            the loop count used at each step for the corresponding sequence.
        """
        self.model.eval()
        if loop_increase_step is None:
            loop_increase_step = max_loops//3#defaulting to 1/3 in case not specified.
        with torch.no_grad():
            # Prepare inputs (for simplicity, only handle batch size 1)
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            if batch_size != 1:
                raise ValueError("Entropix search currently only supports batch size 1")
            
            # Encode spectral data
            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Start with BOS token
            root = EntropixNode(None, None, self.bos_token_id, 0, 0, curr_loops=1)
            current_token = torch.tensor([[self.bos_token_id]], device=self.device)
            
            # Initial step - get logits for the first token
            logits = self.model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens)
            logits = logits[0, -1, :]  # (vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Calculate entropy and varentropy for the first token distribution
            entropy = self.calculate_entropy(log_probs)
            varentropy = self.calculate_varentropy(log_probs)
            
            # Track if we're in a high entropy state from the previous node
            previous_high_entropy = False
            
            # Determine initial action based on entropy and varentropy
            active_nodes = []
            completed_nodes = []
            
            # Low entropy, low varentropy: branch among top-k
            if entropy < entropy_threshold and varentropy < varentropy_threshold:
                # Branch among top-k
                topk_log_probs, topk_indices = log_probs.topk(top_k)
                for i in range(top_k):
                    token_id = topk_indices[i].item()
                    log_prob = topk_log_probs[i].item()
                    active_nodes.append(EntropixNode(None, root, token_id, log_prob, 1, curr_loops=1))
                
                previous_high_entropy = False
            
            # Low entropy, high varentropy: argmax (greedy)
            elif entropy < entropy_threshold and varentropy >= varentropy_threshold:
                # Just take the best token
                token_id = log_probs.argmax().item()
                log_prob = log_probs[token_id].item()
                active_nodes.append(EntropixNode(None, root, token_id, log_prob, 1, curr_loops=1))
                
                previous_high_entropy = False
            
            # High entropy cases - start with looping decision based on initial state
            else:
                # Initial state is high entropy, set flag
                previous_high_entropy = True
                
                # Determine initial loops to request
                if automatic_loop_exit:
                    loops_to_request = max_loops
                else:
                    # In non-automatic mode, start with 1 loop for the first high-entropy token
                    loops_to_request = 1 

                # Apply decoder with looping if needed
                if loops_to_request > 1:
                    logits = self.model.decoder(
                        tgt=current_token, 
                        memory=memory, 
                        nmr_tokens=nmr_tokens, 
                        num_loops=loops_to_request
                    )
                    logits = logits[0, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Recalculate entropy/varentropy after initial loop
                    entropy = self.calculate_entropy(log_probs)
                    varentropy = self.calculate_varentropy(log_probs)

                # Now decide initial action based on potentially updated entropy/varentropy
                if entropy < entropy_threshold and varentropy < varentropy_threshold:
                    # Looping helped, branch among top-k
                    topk_log_probs, topk_indices = log_probs.topk(top_k)
                    for i in range(top_k):
                        token_id = topk_indices[i].item()
                        log_prob = topk_log_probs[i].item()
                        active_nodes.append(EntropixNode(None, root, token_id, log_prob, 1, curr_loops=1))
                    previous_high_entropy = False
                elif entropy < entropy_threshold and varentropy >= varentropy_threshold:
                    # Looping helped, take argmax
                    token_id = log_probs.argmax().item()
                    log_prob = log_probs[token_id].item()
                    active_nodes.append(EntropixNode(None, root, token_id, log_prob, 1, curr_loops=1))
                    previous_high_entropy = False
                else:
                    # Still high entropy, take argmax
                    token_id = log_probs.argmax().item()
                    log_prob = log_probs[token_id].item()
                    # Store the number of loops requested (even if 1) in the node for potential use in next non-automatic step
                    active_nodes.append(EntropixNode(None, root, token_id, log_prob, 1, curr_loops=loops_to_request)) 
                    previous_high_entropy = True

            # Entropix tree search
            for step in range(1, max_len):
                next_active_nodes = []
                
                # Process each active node
                for node in active_nodes:
                    # If this node contains EOS token, add to completed nodes
                    if node.token_id == self.eos_token_id:
                        completed_nodes.append(node)
                        continue
                    
                    # Create sequence for this node and get next token distribution
                    seq = []
                    n = node
                    while n.prev_node:
                        seq.append(n.token_id)
                        n = n.prev_node
                    seq.append(self.bos_token_id)
                    seq.reverse()
                    
                    # Convert to tensor and get model predictions
                    seq_tensor = torch.tensor([seq], device=self.device)
                    
                    # Get the next token distribution (initially without looping)
                    logits = self.model.decoder(tgt=seq_tensor, memory=memory, nmr_tokens=nmr_tokens)
                    logits = logits[0, -1, :]  # (vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Calculate entropy and varentropy
                    entropy = self.calculate_entropy(log_probs)
                    varentropy = self.calculate_varentropy(log_probs)
                    
                    # Determine the number of loops to use
                    loops_for_next_node = 1
                    
                    # Low entropy, low varentropy: branch among top-k
                    if entropy < entropy_threshold and varentropy < varentropy_threshold:
                        # Reset loops to 1 since we're in a low entropy state
                        loops_for_next_node = 1
                        
                        # Branch among top-k
                        topk_log_probs, topk_indices = log_probs.topk(top_k)
                        for i in range(top_k):
                            token_id = topk_indices[i].item()
                            log_prob = topk_log_probs[i].item()
                            next_active_nodes.append(EntropixNode(
                                hidden_state=None,
                                prev_node=node,
                                token_id=token_id,
                                log_prob=node.log_prob + log_prob,
                                length=node.length + 1,
                                curr_loops=loops_for_next_node
                            ))
                        
                        previous_high_entropy = False
                    
                    # Low entropy, high varentropy: argmax (greedy)
                    elif entropy < entropy_threshold and varentropy >= varentropy_threshold:
                        # Reset loops to 1 since we're in a low entropy state
                        loops_for_next_node = 1
                        
                        # Just take the best token
                        token_id = log_probs.argmax().item()
                        log_prob = log_probs[token_id].item()
                        next_active_nodes.append(EntropixNode(
                            hidden_state=None,
                            prev_node=node,
                            token_id=token_id,
                            log_prob=node.log_prob + log_prob,
                            length=node.length + 1,
                            curr_loops=loops_for_next_node
                        ))
                        
                        previous_high_entropy = False
                    
                    # High entropy cases - implement layer looping
                    else:
                        # Determine loops to request based on mode and previous state
                        if automatic_loop_exit:
                            # Always request max_loops, decoder handles early exit
                            loops_to_request = max_loops
                        else:
                            # Non-automatic mode: increase loops if previous state was high entropy
                            if previous_high_entropy:
                                if node.curr_loops >= max_loops:
                                     print(f"Hit maximum loops ({max_loops}) at sequence position {node.length}")
                                loops_to_request = min(node.curr_loops + loop_increase_step, max_loops)
                            else:
                                # Newly entering high entropy state, start with 1 loop request
                                # (Decoder only actually loops if loops_to_request > 1)
                                loops_to_request = 1 
                        
                        # Apply decoder with looping if requested
                        if loops_to_request > 1:
                            # Get logits with layer looping
                            logits = self.model.decoder(
                                tgt=seq_tensor, 
                                memory=memory, 
                                nmr_tokens=nmr_tokens, 
                                num_loops=loops_to_request
                            )
                            logits = logits[0, -1, :]  # (vocab_size)
                            log_probs = torch.log_softmax(logits, dim=-1)
                            
                            # Recalculate entropy and varentropy after looping
                            new_entropy = self.calculate_entropy(log_probs)
                            new_varentropy = self.calculate_varentropy(log_probs)
                            
                            # Re-evaluate strategy based on the new entropy/varentropy values
                            if new_entropy < entropy_threshold and new_varentropy < varentropy_threshold:
                                # The looping helped! Now we have low entropy, low varentropy
                                # So we should branch among top-k
                                topk_log_probs, topk_indices = log_probs.topk(top_k)
                                for i in range(top_k):
                                    token_id = topk_indices[i].item()
                                    log_prob = topk_log_probs[i].item()
                                    next_active_nodes.append(EntropixNode(
                                        hidden_state=None,
                                        prev_node=node,
                                        token_id=token_id,
                                        log_prob=node.log_prob + log_prob,
                                        length=node.length + 1,
                                        curr_loops=1 # Reset loops for next node as entropy resolved
                                    ))
                                # We're no longer in a high entropy state
                                previous_high_entropy = False
                                continue  # Skip to next node in loop
                            elif new_entropy < entropy_threshold and new_varentropy >= varentropy_threshold:
                                # The looping helped! Now we have low entropy, high varentropy
                                # So we should take the argmax
                                token_id = log_probs.argmax().item()
                                log_prob = log_probs[token_id].item()
                                next_active_nodes.append(EntropixNode(
                                    hidden_state=None,
                                    prev_node=node,
                                    token_id=token_id,
                                    log_prob=node.log_prob + log_prob,
                                    length=node.length + 1,
                                    curr_loops=1 # Reset loops for next node as entropy resolved
                                ))
                                # We're no longer in a high entropy state
                                previous_high_entropy = False
                                continue  # Skip to next node in loop
                            # Otherwise, we're still in high entropy (fall through to next block)
                        
                        # Still high entropy (either because loops_to_request was 1, 
                        # or because looping didn't resolve it)
                        # Just take the best token based on the current log_probs 
                        # (which might be from looping or from the initial calculation)
                        token_id = log_probs.argmax().item()
                        log_prob = log_probs[token_id].item()
                        next_active_nodes.append(EntropixNode(
                            hidden_state=None,
                            prev_node=node,
                            token_id=token_id,
                            log_prob=node.log_prob + log_prob,
                            length=node.length + 1,
                            # Store loops requested for potential use in next non-automatic step
                            curr_loops=loops_to_request 
                        ))
                        
                        previous_high_entropy = True
                
                # Update active nodes (keep a reasonable number of nodes)
                active_nodes = sorted(next_active_nodes, reverse=True)[:top_k]
                
                # Early stopping if all branches end or reach maximum length
                if not active_nodes:
                    break
            
            # Combine completed and active nodes
            all_nodes = completed_nodes + active_nodes
            
            # Sort by log probability
            all_nodes.sort(reverse=True)
            
            # Take the top sequences (up to top_k)
            top_nodes = all_nodes[:top_k]
            
            # Reconstruct the sequences and their loop counts
            sequences = []
            loop_counts_per_sequence = []
            for node in top_nodes:
                seq = []
                loops = []
                n = node
                while n.prev_node:
                    seq.append(n.token_id)
                    loops.append(n.curr_loops)
                    n = n.prev_node
                seq.append(self.bos_token_id)
                seq.reverse()
                loops.reverse() # Align loops with sequence steps
                sequences.append(seq)
                loop_counts_per_sequence.append(loops)
            
            # Post-process and decode the sequences
            decoded_sequences = self.postprocess_sequences(sequences)
            return decoded_sequences, loop_counts_per_sequence
    
    def greedy_decode_with_loops(self, nmr_tokens, ir_data, mass_data=None, max_len=128, num_loops=1, loops_representation=False):
        """
        Greedy decoding strategy with layer looping - selects the most probable token at each step,
        but applies the decoder with a specified num_loops parameter at each step.

        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            num_loops: Number of layer looping iterations to apply in the decoder
            loops_representation: Whether to track and return representations across loops

        Returns:
            If loops_representation=False: List of decoded sequences
            If loops_representation=True: Tuple of (List of decoded sequences, List of loop representations)
        """
        self.model.eval()
        with torch.no_grad():
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            current_token = torch.tensor([[self.bos_token_id]] * batch_size, device=self.device)
            generated_sequences = [[self.bos_token_id] for _ in range(batch_size)]
            
            # Initialize the decoder output and track if we need loop representations
            decoder_output = None
            
            for _ in range(max_len):
                # Call decoder with loops_representation parameter
                if loops_representation:
                    logits, loop_reps = self.model.decoder(
                        tgt=current_token, 
                        memory=memory, 
                        nmr_tokens=nmr_tokens, 
                        num_loops=num_loops
                    )
                    # Store loop representations from the first token generation step only
                    if decoder_output is None:
                        decoder_output = loop_reps
                else:
                    logits = self.model.decoder(
                        tgt=current_token, 
                        memory=memory, 
                        nmr_tokens=nmr_tokens, 
                        num_loops=num_loops
                    )
                    
                next_token = logits[:, -1:].argmax(dim=-1)
                for i in range(batch_size):
                    token_id = next_token[i].item()
                    generated_sequences[i].append(token_id)
                    if token_id == self.eos_token_id:
                        break
                current_token = torch.cat((current_token, next_token), dim=1)
            
            # Post-process and return appropriate results
            decoded_sequences = self.postprocess_sequences(generated_sequences)
            
            if loops_representation and decoder_output is not None:
                return decoded_sequences, decoder_output
            else:
                return decoded_sequences

    def greedy_decode_step_by_step(self, nmr_tokens, ir_data, mass_data=None, max_len=128, num_loops=1):
        """
        Greedy decoding that yields the sequence at each step of token generation.
        Assumes batch_size = 1 for simplicity in yielding intermediate sequences.

        Args:
            nmr_tokens: Tokenized NMR data (should be [1, seq_len])
            ir_data: IR data (should be [1, seq_len] or [1, num_features])
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            num_loops: Number of layer looping iterations for the decoder 
                       (passed to model.decoder if it uses this parameter for its internal mechanism).

        Yields:
            torch.Tensor: Tensor of token IDs generated so far at each step (e.g., [bos, t1], [bos, t1, t2], ...)
                         for the first item in the batch.
        """
        self.model.eval()
        with torch.no_grad():
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            if batch_size != 1:
                raise ValueError("greedy_decode_step_by_step currently supports batch_size=1 for simplicity.")

            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            current_token_list = [self.bos_token_id]
            model_input_tokens = torch.tensor([current_token_list], device=self.device)

            yield torch.tensor(current_token_list, device=self.device) # Yield initial BOS token

            for _ in range(max_len):
                # The model.decoder might have its own looping mechanism (num_loops parameter).
                # This is distinct from the autoregressive steps.
                # We pass num_loops here in case the model uses it for its internal processing at each generation step.
                decoder_output = self.model.decoder(
                    tgt=model_input_tokens, 
                    memory=memory, 
                    nmr_tokens=nmr_tokens, 
                    num_loops=num_loops 
                )
                
                # Check if decoder_output is a tuple (logits, potentially other data like loop_reps)
                if isinstance(decoder_output, tuple):
                    logits = decoder_output[0]
                else:
                    logits = decoder_output # Assume it's just logits
                
                next_token_logit = logits[:, -1, :] 
                next_token_id = next_token_logit.argmax(dim=-1).item()
                
                current_token_list.append(next_token_id)
                model_input_tokens = torch.tensor([current_token_list], device=self.device)

                yield torch.tensor(current_token_list, device=self.device)

                if next_token_id == self.eos_token_id:
                    break


# Simple usage example
if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from models.multimodal_to_smiles import MultiModalToSMILESModel
    from models.smiles_tokenizer import SmilesTokenizer
    import json
    import os
    
    parser = argparse.ArgumentParser(description="Run inference with different decoding strategies")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "beam", "sampling", "nucleus"],
                        help="Decoding strategy")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--raw_nmr", type=str, default=None, help="Path to raw NMR spectrum text file")
    parser.add_argument("--raw_ir", type=str, default=None, help="Path to raw IR spectrum text file")
    
    args = parser.parse_args()
    
    # Load configuration
    def load_config(config_path=None):
        # Default minimal configuration (can be extended via a YAML file)
        default_config = {
            'model': {
                'max_seq_length': 512,
                'max_nmr_length': 128,
                'max_memory_length': 128,
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'use_stablemax': False,
                'width_basis': 13
            },
            'data': {
                'tokenized_dir': 'tokenized_baseline/data'
            }
        }
        if config_path is not None:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = update_dict(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d
                update_dict(default_config, custom_config)
        return default_config
    
    config = load_config(args.config)
    
    # Initialize tokenizers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)
    
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1
    
    model = MultiModalToSMILESModel(
        smiles_vocab_size=smiles_vocab_size,
        nmr_vocab_size=nmr_vocab_size,
        max_seq_length=config['model']['max_seq_length'],
        max_nmr_length=config['model']['max_nmr_length'],
        max_memory_length=config['model']['max_memory_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        verbose=False,
        use_stablemax=config['model'].get('use_stablemax', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False)
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Helper functions for loading raw spectra
    def load_raw_spectrum_tokens(file_path, spectral_tokenizer, max_len):
        tokens_list = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        intensity = float(parts[1])
                        # Round intensity to 2 decimals and convert to string token
                        token = str(round(intensity, 2))
                        tokens_list.append(token)
                    except:
                        continue
        token_str = " ".join(tokens_list)
        tokens = token_str.split()
        token_ids = [spectral_tokenizer.get(t, spectral_tokenizer.get("<UNK>")) for t in tokens]
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def load_raw_ir(file_path):
        intensities = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        intensities.append(float(parts[1]))
                    except:
                        continue
        return torch.tensor(intensities, dtype=torch.float32)
    
    # Load raw spectra if provided
    nmr_tokens = None
    ir_data = None
    
    if args.raw_nmr:
        nmr_tokens = load_raw_spectrum_tokens(
            args.raw_nmr, 
            nmr_tokenizer, 
            config['model']['max_nmr_length']
        ).to(device)
    
    if args.raw_ir:
        ir_data = load_raw_ir(args.raw_ir).to(device)
    
    # Create inference class
    inference = ModelInference(model, tokenizer, device)
    
    # Choose strategy
    strategy = DecodingStrategy(args.strategy)
    
    # Run inference
    results = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=strategy,
        max_len=args.max_len,
        beam_width=args.beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Print results
    print(f"\nResults using {strategy.value} decoding:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result}") 