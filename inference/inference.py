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
    BEAM_HYBRID = "beam_hybrid"
    BEAM_AUTO_LOOP = "beam_auto_loop"


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
    def __init__(self, hidden_state, prev_node, token_id, log_prob, length, curr_loops=1, was_gen_path_high_entropy: bool = False):
        """
        Initialize an Entropix search node.
        
        Args:
            hidden_state: The hidden state (not used in current implementation)
            prev_node: The previous node in the sequence
            token_id: The token ID for this node
            log_prob: The cumulative log probability of the sequence so far
            length: The length of the sequence so far
            curr_loops: Current number of loops used for this node
            was_gen_path_high_entropy: True if the generation path leading to this node ended in high entropy.
        """
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
        self.curr_loops = curr_loops  # Track how many loops were used
        self.was_gen_path_high_entropy = was_gen_path_high_entropy
    
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
        entropy_threshold_beam_hybrid: float = 1.0,
        beam_hybrid_max_loops: int = 3,
        auto_loop_threshold_beam_auto_loop: Optional[float] = None,
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
            max_loops: Maximum number of times to loop the middle layer for Entropix
            automatic_loop_exit: Whether to automatically exit loops based on entropy in Entropix
            automatic_loop_exit_threshold: Threshold for automatic loop exit in Entropix
            loop_increase_step: Amount to increase loop count by when in high-entropy (non-automatic mode) in Entropix
            entropy_threshold_beam_hybrid: Entropy threshold for Beam Hybrid strategy to trigger looping.
            beam_hybrid_max_loops: Number of loops for Beam Hybrid strategy when entropy is high.
            auto_loop_threshold_beam_auto_loop: Specific convergence threshold for BEAM_AUTO_LOOP. If None, model's default is used.
            
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
        elif strategy == DecodingStrategy.BEAM_HYBRID:
            return self.beam_search_hybrid(
                nmr_tokens=nmr_tokens, 
                ir_data=ir_data, 
                mass_data=mass_data, 
                max_len=max_len, 
                beam_width=beam_width,
                length_penalty=length_penalty,
                entropy_threshold=kwargs.get("entropy_threshold_beam_hybrid", entropy_threshold_beam_hybrid),
                max_loops_on_high_entropy=kwargs.get("beam_hybrid_max_loops", beam_hybrid_max_loops)
            )
        elif strategy == DecodingStrategy.BEAM_AUTO_LOOP:
            return self.beam_search_auto_loop(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                mass_data=mass_data,
                max_len=max_len,
                beam_width=beam_width,
                length_penalty=length_penalty,
                auto_loop_threshold=(auto_loop_threshold_beam_auto_loop 
                                     if auto_loop_threshold_beam_auto_loop is not None 
                                     else self.model.decoder.automatic_loop_exit_threshold)
            )
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
                logits = self.model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens,num_loops=1)
                
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
                nodes.append(BeamSearchNode(
                    hidden_state=None,
                    prev_node=node,
                    token_id=token_id,
                    log_prob=log_prob,
                    length=1
                ))
            
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
    
    def _get_logits_and_metrics(self, seq_tensor: torch.Tensor, memory: torch.Tensor, nmr_tokens: Optional[torch.Tensor], num_loops: int = 1) -> Tuple[torch.Tensor, float, float]:
        """
        Helper to get logits from the model and calculate entropy/varentropy.
        """
        logits = self.model.decoder(tgt=seq_tensor, memory=memory, nmr_tokens=nmr_tokens, num_loops=num_loops)
        logits = logits[0, -1, :]  # (vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = self.calculate_entropy(log_probs)
        varentropy = self.calculate_varentropy(log_probs)
        return log_probs, entropy, varentropy

    def _entropix_create_node(self, parent_node: Optional[EntropixNode], token_id: int, log_prob: float, length: int, curr_loops: int, was_gen_path_high_entropy: bool) -> EntropixNode:
        """
        Helper to create an EntropixNode.
        """
        cumulative_log_prob = (parent_node.log_prob if parent_node else 0) + log_prob
        return EntropixNode(
            hidden_state=None,
            prev_node=parent_node,
            token_id=token_id,
            log_prob=cumulative_log_prob,
            length=length,
            curr_loops=curr_loops,
            was_gen_path_high_entropy=was_gen_path_high_entropy
        )

    def _entropix_branch_action(
        self, 
        parent_node: Optional[EntropixNode], 
        log_probs: torch.Tensor, 
        k_branch: int,
        epsilon: float, 
        is_initial_step: bool
    ) -> List[EntropixNode]:
        """
        Handles low entropy, low varentropy: branch among top-k (potentially filtered).
        """
        nodes = []
        topk_log_probs, topk_indices = log_probs.topk(k_branch)
        
        if is_initial_step: # Apply epsilon filtering only for the very first token generation
            average_logprobs = torch.mean(topk_log_probs)
            for i in range(k_branch):
                token_id = topk_indices[i].item()
                lp = topk_log_probs[i].item()
                if abs(lp - average_logprobs) < epsilon:
                    nodes.append(self._entropix_create_node(parent_node, token_id, lp, (parent_node.length if parent_node else 0) + 1, 1, False))
        else:
            for i in range(k_branch):
                token_id = topk_indices[i].item()
                lp = topk_log_probs[i].item()
                nodes.append(self._entropix_create_node(parent_node, token_id, lp, (parent_node.length if parent_node else 0) + 1, 1, False))
        return nodes

    def _entropix_argmax_action(
        self, 
        parent_node: Optional[EntropixNode], 
        log_probs: torch.Tensor, 
        loops_applied: int, 
        path_is_high_entropy: bool
    ) -> List[EntropixNode]:
        """
        Handles argmax selection (e.g., low entropy, high varentropy or post-looping high entropy).
        """
        token_id = log_probs.argmax().item()
        lp = log_probs[token_id].item()
        return [self._entropix_create_node(parent_node, token_id, lp, (parent_node.length if parent_node else 0) + 1, loops_applied, path_is_high_entropy)]

    def _entropix_handle_high_entropy_step(
        self,
        parent_node: EntropixNode, # Can be root node for initial step
        seq_tensor: torch.Tensor,
        memory: torch.Tensor,
        nmr_tokens: Optional[torch.Tensor],
        initial_log_probs: torch.Tensor, 
        initial_entropy: float,
        initial_varentropy: float, # Added initial_varentropy
        max_loops: int,
        automatic_loop_exit: bool,
        loop_increase_step: int,
    ) -> Tuple[torch.Tensor, float, float, int]: # Returns: log_probs, entropy, varentropy, loops_applied
        """
        Manages high entropy situations by deciding on and applying looping.
        Returns the updated log_probs, entropy, varentropy, and the number of loops applied in this attempt.
        It does NOT create nodes.
        """
        loops_to_request = 1
        current_loops_of_parent = parent_node.curr_loops if parent_node else 1

        if automatic_loop_exit:
            loops_to_request = max_loops
        else: # Non-automatic mode
            if parent_node and parent_node.was_gen_path_high_entropy: # If previous step was already high-entropy
                loops_to_request = min(current_loops_of_parent + loop_increase_step, max_loops)
                if current_loops_of_parent >= max_loops:
                    # print(f"Hit maximum loops ({max_loops}) at sequence position {parent_node.length}. No further looping attempted in this step.")
                    # Return original high-entropy state as no new looping is productive here
                    return initial_log_probs, initial_entropy, initial_varentropy, current_loops_of_parent
            else: # Newly entering high entropy state, or parent was not high-entropy
                loops_to_request = 1 # Default to 1, meaning we might try more if conditions below are met for first entry
                # If it's the first time encountering high entropy for this path, 
                # and we are not in automatic_loop_exit, we might want to start with more than 1 loop.
                # However, the current logic effectively means if loops_to_request is 1 here, no *additional* looping is forced by this function alone.
                # The design is that if loops_to_request > 1, we explicitly loop.
                # If loops_to_request is 1, we return the initial state, and the main loop decides based on that.
                # Let's refine: if it's a new high entropy state, and not automatic, it should at least try base looping if max_loops > 1
                # For simplicity, if loops_to_request calculated so far is 1, and it's a high entropy state, we will return initial state.
                # The main function will then decide. This makes this function purely about *increasing* loops if parent was high-E.

        if loops_to_request > 1: # Only perform new looping if more than 1 loop is requested
            # print(f"High Entropy: Attempting {loops_to_request} loops for node after token {parent_node.token_id if parent_node else 'BOS'} (len {parent_node.length if parent_node else 0})")
            looped_log_probs, new_entropy, new_varentropy = self._get_logits_and_metrics(
                seq_tensor, memory, nmr_tokens, num_loops=loops_to_request
            )
            return looped_log_probs, new_entropy, new_varentropy, loops_to_request
        else:
            # No conditions met to increase loops or already at max for non-automatic.
            # Return the initial high-entropy state and the parent's loop count (or 1 if no parent/root).
            # This signifies that this function didn't apply *new* looping.
            # print(f"High Entropy: Not attempting further loops (req: {loops_to_request}, parent loops: {current_loops_of_parent}). Using initial metrics for node after token {parent_node.token_id if parent_node else 'BOS'}")
            return initial_log_probs, initial_entropy, initial_varentropy, current_loops_of_parent
    
    def entropix_decode(self, nmr_tokens, ir_data, mass_data=None, max_len=128, top_k=5, 
                       entropy_threshold=1.0, varentropy_threshold=0.5, max_loops=3, 
                       automatic_loop_exit=False, automatic_loop_exit_threshold=0.01, # Note: automatic_loop_exit_threshold not directly used
                       loop_increase_step=None, epsilon=1000, k_branch: int = 2): # User changed epsilon, added k_branch
        """
        Entropix tree search - uses entropy and varentropy to make branching decisions.
        Refactored for modular actions.
        """
        self.model.eval()
        if loop_increase_step is None:
            loop_increase_step = max_loops // 3 if max_loops > 0 else 1 
            if loop_increase_step == 0: loop_increase_step = 1

        with torch.no_grad():
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            if batch_size != 1:
                raise ValueError("Entropix search currently only supports batch size 1")

            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Root node represents BOS, length 0, 1 loop (nominal), not high entropy path
            root = EntropixNode(None, None, self.bos_token_id, 0, 0, curr_loops=1, was_gen_path_high_entropy=False)
            
            active_nodes: List[EntropixNode] = []
            completed_nodes: List[EntropixNode] = []

            # --- Initial step (from BOS to first token) ---
            initial_seq_tensor = torch.tensor([[self.bos_token_id]], device=self.device)
            current_log_probs, current_entropy, current_varentropy = self._get_logits_and_metrics(
                initial_seq_tensor, memory, nmr_tokens, num_loops=1
            )
            
            loops_applied_in_current_attempt = 1 # Initially, we tried 1 loop
            # path_leading_to_children_is_high_entropy = False # Set based on conditions

            if current_entropy >= entropy_threshold:
                # print(f"Initial step is High Entropy: E={current_entropy:.2f}, V={current_varentropy:.2f}. Handling...")
                current_log_probs, current_entropy, current_varentropy, loops_applied_in_current_attempt = self._entropix_handle_high_entropy_step(
                    root, initial_seq_tensor, memory, nmr_tokens, 
                    current_log_probs, current_entropy, current_varentropy, 
                    max_loops, automatic_loop_exit, loop_increase_step
                )
                # print(f"After handling initial HE: New E={current_entropy:.2f}, New V={current_varentropy:.2f}, Loops applied: {loops_applied_in_current_attempt}")

            # Now, make decision based on potentially updated current_entropy/varentropy
            generated_nodes_for_initial_step: List[EntropixNode] = []
            initial_path_resolved_to_low_entropy = (current_entropy < entropy_threshold)

            if initial_path_resolved_to_low_entropy:
                loops_for_children_creation = 1 # Reset loops if entropy resolved
                if current_varentropy < varentropy_threshold: # Low E, Low V
                    # print("Initial: Low E, Low V -> Branching")
                    generated_nodes_for_initial_step = self._entropix_branch_action(root, current_log_probs, k_branch, epsilon, is_initial_step=True)
                else: # Low E, High V
                    # print("Initial: Low E, High V -> Argmax")
                    generated_nodes_for_initial_step = self._entropix_argmax_action(root, current_log_probs, loops_for_children_creation, False)
                for node in generated_nodes_for_initial_step:
                    node.was_gen_path_high_entropy = False # Path for these children is low entropy
            else: # Still High Entropy (E >= threshold)
                if not automatic_loop_exit and loops_applied_in_current_attempt < max_loops:
                    # STALL: Root node (BOS) persists with updated loop state. No child token generated yet.
                    # print(f"Initial: STALL. BOS persists with loops {loops_applied_in_current_attempt}, still High E.")
                    root.curr_loops = loops_applied_in_current_attempt
                    root.was_gen_path_high_entropy = True
                    # Add root itself to active_nodes, it hasn't generated a child yet.
                    # Its length is 0. It will be processed by the main loop.
                    generated_nodes_for_initial_step = [root] 
                else:
                    # MAX LOOPS reached for BOS or AUTO_MODE: Argmax despite high entropy.
                    # print(f"Initial: Still High E (E={current_entropy:.2f}). Max loops or auto. Argmax with loops {loops_applied_in_current_attempt}")
                    generated_nodes_for_initial_step = self._entropix_argmax_action(root, current_log_probs, loops_applied_in_current_attempt, True)
                    for node in generated_nodes_for_initial_step: # Should be one node
                        node.was_gen_path_high_entropy = True # Path for this child is high entropy
            
            active_nodes = generated_nodes_for_initial_step
            # Note: if root was added, active_nodes contains a node of length 0.

            # --- Entropix tree search main loop ---
            for step in range(1, max_len): 
                if not active_nodes:
                    break
                
                next_active_nodes: List[EntropixNode] = []
                
                for current_node in active_nodes:
                    if current_node.token_id == self.eos_token_id and current_node.length > 0: # EOS is valid if not on a stalled BOS node
                        completed_nodes.append(current_node)
                        continue
                    
                    # If current_node is BOS (length 0) and it's here, it means it stalled in initial step
                    # or it stalled in a previous iteration of this loop.
                    # Sequence for BOS is just BOS itself.
                    if current_node.length == 0 and current_node.token_id == self.bos_token_id:
                        seq_tensor = torch.tensor([[self.bos_token_id]], device=self.device)
                    else: # Reconstruct sequence for a normal node that has generated tokens
                        seq_list = []
                        temp_n = current_node
                        while temp_n is not None and temp_n.prev_node is not None: 
                            seq_list.append(temp_n.token_id)
                            temp_n = temp_n.prev_node
                        seq_list.append(self.bos_token_id)
                        seq_list.reverse()
                        seq_tensor = torch.tensor([seq_list], device=self.device)

                    # Get metrics for the next token distribution (always with 1 loop initially for this step's consideration)
                    # However, if current_node.curr_loops > 1 (due to stalling), it implies we should use that many loops.
                    # The _get_logits_and_metrics should be called with current_node.curr_loops if we are re-evaluating a stalled node.
                    # The _entropix_handle_high_entropy_step takes current_node as parent and uses its state.
                    
                    # Initial metrics for this step, based on current_node's *current* loop state if it was high-E previously,
                    # or 1 loop if it was low-E previously.
                    # loops_to_use_for_initial_metric = current_node.curr_loops if current_node.was_gen_path_high_entropy else 1
                    # No, always start with 1 loop for the *candidate* next token, then handler uses parent state.
                    iter_log_probs, iter_entropy, iter_varentropy = self._get_logits_and_metrics(
                        seq_tensor, memory, nmr_tokens, num_loops=1 # Base attempt for this step
                    )
                    
                    current_log_probs_for_step = iter_log_probs
                    current_entropy_for_step = iter_entropy
                    current_varentropy_for_step = iter_varentropy
                    loops_applied_by_handler_or_initial = current_node.curr_loops # Start with parent's current loop count as baseline for handler
                    if not current_node.was_gen_path_high_entropy: # If parent was low-E, this step's attempt starts with 1 loop
                         loops_applied_by_handler_or_initial = 1

                    if current_entropy_for_step >= entropy_threshold:
                        # print(f"Step {step}, Node {current_node.token_id} (len {current_node.length}, parent loops {current_node.curr_loops}, parent HE {current_node.was_gen_path_high_entropy}): High Entropy E={current_entropy_for_step:.2f}. Handling...")
                        current_log_probs_for_step, current_entropy_for_step, current_varentropy_for_step, loops_applied_by_handler_or_initial = \
                            self._entropix_handle_high_entropy_step(
                                current_node, seq_tensor, memory, nmr_tokens, 
                                current_log_probs_for_step, current_entropy_for_step, current_varentropy_for_step,
                                max_loops, automatic_loop_exit, loop_increase_step
                            )
                        # print(f"After HE handle: New E={current_entropy_for_step:.2f}, Loops applied: {loops_applied_by_handler_or_initial}")

                    # Make decision for generating children based on potentially updated state
                    generated_children_this_iter: List[EntropixNode] = []
                    path_for_children_is_low_entropy = (current_entropy_for_step < entropy_threshold)

                    if path_for_children_is_low_entropy:
                        # print(f"Step {step}, Node {current_node.token_id}: Low E. Creating child(ren).")
                        loops_for_low_e_child = 1 
                        if current_varentropy_for_step < varentropy_threshold: # Low E, Low V
                            generated_children_this_iter = self._entropix_branch_action(current_node, current_log_probs_for_step, k_branch, epsilon, is_initial_step=False)
                        else: # Low E, High V
                            generated_children_this_iter = self._entropix_argmax_action(current_node, current_log_probs_for_step, loops_for_low_e_child, False)
                        for ngn_node in generated_children_this_iter:
                            ngn_node.was_gen_path_high_entropy = False
                    else: # Still High Entropy path (E >= threshold)
                        if not automatic_loop_exit and loops_applied_by_handler_or_initial < max_loops:
                            # STALL: current_node persists with updated loop state. No child token generated for this position yet.
                            # print(f"Step {step}, Node {current_node.token_id}: STALL. Persisting with loops {loops_applied_by_handler_or_initial}, still High E.")
                            current_node.curr_loops = loops_applied_by_handler_or_initial
                            current_node.was_gen_path_high_entropy = True
                            generated_children_this_iter = [current_node] # Add current_node itself to be carried over
                        else:
                            # MAX LOOPS reached for this path or AUTO_MODE: Argmax despite high entropy.
                            # print(f"Step {step}, Node {current_node.token_id}: Max loops or auto. Argmax with loops {loops_applied_by_handler_or_initial}, still High E.")
                            generated_children_this_iter = self._entropix_argmax_action(current_node, current_log_probs_for_step, loops_applied_by_handler_or_initial, True)
                            for ngn_node in generated_children_this_iter: # Should be one node
                                ngn_node.was_gen_path_high_entropy = True 
                    
                    next_active_nodes.extend(generated_children_this_iter)

                active_nodes = sorted(list(set(next_active_nodes)), reverse=True)[:top_k]
                
                if not active_nodes and not completed_nodes: # All paths died out
                    break
            
            all_nodes = completed_nodes + active_nodes
            all_nodes.sort(reverse=True)
            top_nodes = all_nodes[:top_k]
            
            sequences = []
            loop_counts_per_sequence = []
            for node in top_nodes:
                seq = []
                loops = []
                n = node
                # prev_node for root is None, so we don't add BOS token here
                while n is not None and n.prev_node is not None: 
                    seq.append(n.token_id)
                    # Store the loops that *led* to this token, so from current node
                    loops.append(n.curr_loops) 
                    n = n.prev_node
                # Add BOS if the sequence is not empty or if it's the only token (root itself, though unlikely to be a top_node)
                if n is not None and n.token_id == self.bos_token_id:
                     seq.append(self.bos_token_id)
                # if not seq and n is not None and n.token_id == self.bos_token_id: # Handle case where only BOS is considered (e.g. max_len=0)
                #     seq.append(self.bos_token_id)

                seq.reverse()
                loops.reverse()
                sequences.append(seq)
                loop_counts_per_sequence.append(loops)
            
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

    def beam_search_hybrid(self, nmr_tokens, ir_data, mass_data=None, max_len=128, beam_width=5, length_penalty=1.0, entropy_threshold: float = 1.0, max_loops_on_high_entropy: int = 3):
        """
        Beam search decoding with entropy-triggered looping.
        Maintains top-k candidate sequences, and if entropy at a step is high,
        it re-calculates logits with additional decoder loops.

        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            beam_width: Number of candidate sequences to maintain
            length_penalty: Penalty factor for sequence length (used in BeamSearchNode.eval)
            entropy_threshold: Entropy threshold to trigger looping.
            max_loops_on_high_entropy: Number of loops to apply when entropy is high.
            
        Returns:
            List of decoded sequences
        """
        self.model.eval()
        with torch.no_grad():
            (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
            if batch_size != 1:
                raise ValueError("Beam search hybrid currently only supports batch size 1")

            memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
            
            # Start with BOS token
            start_node_placeholder = BeamSearchNode(None, None, self.bos_token_id, 0, 0) # Used as prev_node for first real tokens
            
            # Initial step - get log_probs for the first token
            initial_seq_tensor = torch.tensor([[self.bos_token_id]], device=self.device)
            
            current_log_probs, current_entropy, _ = self._get_logits_and_metrics(
                initial_seq_tensor, memory, nmr_tokens, num_loops=1 # Start with 1 loop
            )

            if current_entropy >= entropy_threshold:
                # print(f"BeamHybrid Initial: High Entropy E={current_entropy:.2f}. Iteratively looping up to {max_loops_on_high_entropy} times.")
                for num_loops_attempt in range(2, max_loops_on_high_entropy + 1):
                    # print(f"  Attempting {num_loops_attempt} loops...")
                    temp_log_probs, temp_entropy, _ = self._get_logits_and_metrics(
                        initial_seq_tensor, memory, nmr_tokens, num_loops=num_loops_attempt
                    )
                    current_log_probs = temp_log_probs # Always update to the latest attempt
                    current_entropy = temp_entropy
                    if current_entropy < entropy_threshold:
                        # print(f"    Low entropy E={current_entropy:.2f} achieved with {num_loops_attempt} loops.")
                        break # Found low entropy, use these log_probs
                # If loop finishes and entropy is still high, current_log_probs is from max_loops_on_high_entropy
            
            # Get top beam_width candidates
            topk_log_probs, topk_indices = current_log_probs.topk(beam_width)
            
            # Initialize beam
            nodes = []
            for i in range(beam_width):
                token_id = topk_indices[i].item()
                log_prob = topk_log_probs[i].item()
                nodes.append(BeamSearchNode(
                    hidden_state=None,
                    prev_node=start_node_placeholder, # First actual tokens link to this
                    token_id=token_id,
                    log_prob=log_prob, # Initial log_prob is from the model directly
                    length=1
                ))
            
            # Keep track of sequences that have reached EOS
            endnodes = []
            
            # Beam search loop
            for step in range(2, max_len + 1):
                if len(endnodes) >= beam_width and beam_width > 0 : # Check beam_width > 0 to avoid issues if beam_width is 0
                    break
                
                candidates = []
                
                for node_in_beam in nodes:
                    if node_in_beam.token_id == self.eos_token_id:
                        endnodes.append(node_in_beam)
                        continue
                    
                    # Reconstruct sequence for this node
                    seq_list = []
                    temp_n = node_in_beam
                    while temp_n.prev_node: # Stop before the placeholder start_node_placeholder
                        seq_list.append(temp_n.token_id)
                        temp_n = temp_n.prev_node
                    # BOS was part of initial_seq_tensor, effectively.
                    # The `node_in_beam` itself contains the first *actual* token if length is 1.
                    # So, if node.length is 1, seq_list contains [token_id_1]. We need [BOS, token_id_1] for model.
                    # The reconstruction should always yield the full sequence needed by the model.
                    seq_list.append(self.bos_token_id) 
                    seq_list.reverse()
                    
                    seq_tensor = torch.tensor([seq_list], device=self.device)
                    
                    # Get initial log_probs (with 1 loop) for expanding this node
                    current_log_probs_for_node, current_entropy_for_node, _ = self._get_logits_and_metrics(
                        seq_tensor, memory, nmr_tokens, num_loops=1 # Start with 1 loop
                    )

                    if current_entropy_for_node >= entropy_threshold:
                        # print(f"BeamHybrid Step {step}: High Entropy E={current_entropy_for_node:.2f} for token {node_in_beam.token_id}. Iteratively looping up to {max_loops_on_high_entropy} times.")
                        for num_loops_attempt in range(2, max_loops_on_high_entropy + 1):
                            # print(f"  Attempting {num_loops_attempt} loops...")
                            temp_log_probs, temp_entropy, _ = self._get_logits_and_metrics(
                                seq_tensor, memory, nmr_tokens, num_loops=num_loops_attempt
                            )
                            current_log_probs_for_node = temp_log_probs # Always update to the latest attempt
                            current_entropy_for_node = temp_entropy # Update entropy for the check
                            if current_entropy_for_node < entropy_threshold:
                                # print(f"    Low entropy E={current_entropy_for_node:.2f} achieved with {num_loops_attempt} loops.")
                                break # Found low entropy, use these log_probs
                        # If loop finishes, current_log_probs_for_node is from max_loops_on_high_entropy
                    
                    # Get top beam_width candidates and add to candidate list
                    topk_log_probs_cand, topk_indices_cand = current_log_probs_for_node.topk(beam_width)
                    for i in range(beam_width):
                        token_id = topk_indices_cand[i].item()
                        log_prob = topk_log_probs_cand[i].item()
                        candidates.append(BeamSearchNode(
                            hidden_state=None,
                            prev_node=node_in_beam,
                            token_id=token_id,
                            log_prob=node_in_beam.log_prob + log_prob, # Accumulate log_prob
                            length=node_in_beam.length + 1
                        ))
                
                if not candidates and not endnodes: # No new candidates and no finished sequences
                    break


                # Add completed nodes from this step to all endnodes, then filter from active beam
                active_nodes_after_eos_check = [n for n in nodes if n.token_id != self.eos_token_id]
                
                # If all nodes in the beam ended in EOS, and we don't have enough endnodes,
                # this check might be too aggressive if candidates is empty.
                # The primary check is len(endnodes) >= beam_width at the start of the loop.
                # If candidates is empty, it means all paths from active_nodes_after_eos_check died out or ended.
                if not candidates and len(endnodes) < beam_width : # if beam_width is 0, this is false
                     # All active paths died, take what we have in endnodes
                     # (or if endnodes is also empty, then we have nothing)
                     break


                # Sort all candidates (newly generated) and select top beam_width
                # Combine with existing endnodes only if we are forced to terminate early.
                # The primary list for next iteration is 'nodes'
                if candidates:
                    candidates.sort(key=lambda x: x.eval(alpha=length_penalty), reverse=True)
                    nodes = candidates[:beam_width]
                elif len(endnodes) < beam_width and beam_width > 0 : # No candidates, not enough endnodes
                    nodes = [] # Stop search
                else: # No candidates, but enough endnodes or beam_width is 0
                    nodes = [] # Stop search

                # If after pruning, nodes is empty, and we have some endnodes, we can stop.
                if not nodes and endnodes:
                    break


            # Final selection from endnodes or current beam if not enough finished sequences
            final_candidates = endnodes
            if len(endnodes) < beam_width and beam_width > 0:
                # Add best nodes from the current beam if they are better or fill up the beam
                # Ensure nodes are sorted by score
                sorted_active_nodes = sorted(nodes, key=lambda x: x.eval(alpha=length_penalty), reverse=True)
                final_candidates.extend(sorted_active_nodes)

            # Sort all final candidates by score
            final_candidates.sort(key=lambda x: x.eval(alpha=length_penalty), reverse=True)
            
            # Reconstruct the sequences
            sequences = []
            for node in final_candidates[:beam_width]: # Get top beam_width sequences
                seq = []
                n = node
                # Traverse back until the node whose prev_node is the placeholder (or None if node is placeholder itself)
                while n is not None and n.prev_node is not None: 
                    seq.append(n.token_id)
                    n = n.prev_node
                # After the loop, n is the start_node_placeholder (which holds BOS_TOKEN_ID)
                # or n is None if the loop was never entered (e.g. if node itself was start_node_placeholder - unlikely here)
                # Or if the node was directly created from start_node_placeholder, n will be start_node_placeholder.
                if n is not None and n.token_id == self.bos_token_id: # Add BOS token
                    seq.append(self.bos_token_id)
                
                seq.reverse()
                sequences.append(seq)

            return self.postprocess_sequences(sequences)

    def beam_search_auto_loop(self, nmr_tokens, ir_data, mass_data=None, max_len=128, beam_width=5, length_penalty=1.0, auto_loop_threshold: Optional[float] = None):
        """
        Beam search decoding that leverages the decoder's internal automatic loop exit mechanism.
        The decoder is instructed to loop up to its max_loops, but will exit early if
        its internal convergence criteria (automatic_loop_exit=True) are met.

        Args:
            nmr_tokens: Tokenized NMR data
            ir_data: IR spectral data
            mass_data: Mass spectrometry data
            max_len: Maximum sequence length
            beam_width: Number of candidate sequences to maintain
            length_penalty: Penalty factor for sequence length (used in BeamSearchNode.eval)
            auto_loop_threshold: Specific convergence threshold for BEAM_AUTO_LOOP. If None, model's default is used.
            
        Returns:
            List of decoded sequences
        """
        self.model.eval()
        original_auto_loop_exit_state = self.model.decoder.automatic_loop_exit
        original_auto_loop_threshold = self.model.decoder.automatic_loop_exit_threshold # Store original threshold
        
        self.model.decoder.automatic_loop_exit = True # Ensure it's enabled for this strategy
        if auto_loop_threshold is not None:
            self.model.decoder.automatic_loop_exit_threshold = auto_loop_threshold # Set to specific threshold for this strategy
        
        try:
            with torch.no_grad():
                (nmr_tokens, ir_data, mass_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data, mass_data)
                if batch_size != 1:
                    raise ValueError("Beam search auto_loop currently only supports batch size 1")

                memory = self.encode_inputs(nmr_tokens, ir_data, mass_data)
                
                start_node_placeholder = BeamSearchNode(None, None, self.bos_token_id, 0, 0)
                
                initial_seq_tensor = torch.tensor([[self.bos_token_id]], device=self.device)
                
                # Instruct decoder to loop up to its max_loops; automatic_loop_exit (if True in model) will handle early stop.
                current_log_probs, _, _ = self._get_logits_and_metrics(
                    initial_seq_tensor, memory, nmr_tokens, num_loops=self.model.decoder.max_loops
                )
                
                topk_log_probs, topk_indices = current_log_probs.topk(beam_width)
                
                nodes = []
                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    log_prob = topk_log_probs[i].item()
                    nodes.append(BeamSearchNode(
                        hidden_state=None,
                        prev_node=start_node_placeholder, 
                        token_id=token_id,
                        log_prob=log_prob, 
                        length=1
                    ))
                
                endnodes = []
                
                for step in range(2, max_len + 1):
                    if len(endnodes) >= beam_width and beam_width > 0:
                        break
                    
                    candidates = []
                    
                    for node_in_beam in nodes:
                        if node_in_beam.token_id == self.eos_token_id:
                            endnodes.append(node_in_beam)
                            continue
                        
                        seq_list = []
                        temp_n = node_in_beam
                        while temp_n.prev_node:
                            seq_list.append(temp_n.token_id)
                            temp_n = temp_n.prev_node
                        seq_list.append(self.bos_token_id) 
                        seq_list.reverse()
                        
                        seq_tensor = torch.tensor([seq_list], device=self.device)
                        
                        # Instruct decoder to loop up to its max_loops for node expansion.
                        current_log_probs_for_node, _, _ = self._get_logits_and_metrics(
                            seq_tensor, memory, nmr_tokens, num_loops=self.model.decoder.max_loops
                        )
                        
                        topk_log_probs_cand, topk_indices_cand = current_log_probs_for_node.topk(beam_width)
                        for i in range(beam_width):
                            token_id = topk_indices_cand[i].item()
                            log_prob = topk_log_probs_cand[i].item()
                            candidates.append(BeamSearchNode(
                                hidden_state=None,
                                prev_node=node_in_beam,
                                token_id=token_id,
                                log_prob=node_in_beam.log_prob + log_prob, 
                                length=node_in_beam.length + 1
                            ))
                    
                    if not candidates and not endnodes:
                        break

                    active_nodes_after_eos_check = [n for n in nodes if n.token_id != self.eos_token_id]
                    
                    if not candidates and len(endnodes) < beam_width:
                         break

                    if candidates:
                        candidates.sort(key=lambda x: x.eval(alpha=length_penalty), reverse=True)
                        nodes = candidates[:beam_width]
                    elif len(endnodes) < beam_width and beam_width > 0:
                        nodes = [] 
                    else:
                        nodes = []

                    if not nodes and endnodes:
                        break

                final_candidates = endnodes
                if len(endnodes) < beam_width and beam_width > 0:
                    sorted_active_nodes = sorted(nodes, key=lambda x: x.eval(alpha=length_penalty), reverse=True)
                    final_candidates.extend(sorted_active_nodes)

                final_candidates.sort(key=lambda x: x.eval(alpha=length_penalty), reverse=True)
                
                sequences = []
                for node in final_candidates[:beam_width]:
                    seq = []
                    n = node
                    while n is not None and n.prev_node is not None: 
                        seq.append(n.token_id)
                        n = n.prev_node
                    if n is not None and n.token_id == self.bos_token_id:
                        seq.append(self.bos_token_id)
                    
                    seq.reverse()
                    sequences.append(seq)

                return self.postprocess_sequences(sequences)
        finally:
            self.model.decoder.automatic_loop_exit = original_auto_loop_exit_state # Restore original state
            self.model.decoder.automatic_loop_exit_threshold = original_auto_loop_threshold # Restore original threshold


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