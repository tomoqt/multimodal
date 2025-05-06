import torch
import numpy as np
from inference.inference import ModelInference, DecodingStrategy, EntropixNode


def beam_search_tree(
    m_inf: ModelInference,
    nmr_tokens: torch.Tensor,
    ir_data: torch.Tensor = None,
    beam_width: int = 5,
    max_len: int = 50,
):
    """
    Generator that yields the beam-search active set at each decoding step.
    Yields a list of tuples: (token_id_list, cumulative_log_prob).
    """
    m_inf.model.eval()
    with torch.no_grad():
        # Prepare inputs and memory
        (nmr_tokens_b, ir_data_b, _), _ = m_inf.prepare_inputs(nmr_tokens, ir_data, None)
        memory = m_inf.encode_inputs(nmr_tokens_b, ir_data_b, None)

        # Initialize beam with BOS
        seqs = [[m_inf.bos_token_id]]
        scores = [0.0]
        yield list(zip(seqs, scores))

        for step in range(max_len):
            candidates = []
            for s, sc in zip(seqs, scores):
                input_ids = torch.tensor([s], device=m_inf.device)
                # Decoder forward
                decoder_out = m_inf.model.decoder(
                    tgt=input_ids,
                    memory=memory,
                    nmr_tokens=nmr_tokens_b
                )
                # Log-softmax on last token
                logp = torch.log_softmax(decoder_out[0, -1], dim=-1)
                # Top-k extensions
                topk_vals, topk_ids = torch.topk(logp, beam_width)
                for lp, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
                    candidates.append((s + [int(tid)], sc + float(lp)))
            # Prune beam
            candidates.sort(key=lambda x: x[1], reverse=True)
            seqs, scores = zip(*candidates[:beam_width])
            seqs = [list(x) for x in seqs]
            scores = list(scores)
            yield list(zip(seqs, scores))
            # Stop if all beams ended with EOS
            if all(s[-1] == m_inf.eos_token_id for s in seqs):
                break


def entropix_tree(
    m_inf: ModelInference,
    nmr_tokens: torch.Tensor,
    ir_data: torch.Tensor = None,
    top_k: int = 5,
    entropy_threshold: float = 0.6939,
    varentropy_threshold: float = 1.3781,
    max_loops: int = 3,
    max_len: int = 50,
):
    """Generator that yields the active Entropix node set at each decoding step."""
    m_inf.model.eval()
    with torch.no_grad():
        # Prepare inputs and memory
        (nmr_tokens_b, ir_data_b, _), _ = m_inf.prepare_inputs(nmr_tokens, ir_data, None)
        memory = m_inf.encode_inputs(nmr_tokens_b, ir_data_b, None)
        # Initialize root node
        root = EntropixNode(None, None, m_inf.bos_token_id, 0.0, 0, curr_loops=1)
        active_nodes = [root]
        previous_high_entropy = False
        # Iterate over generation steps
        for step in range(max_len):
            next_active = []
            # Expand each active node
            for node in active_nodes:
                # Preserve completed EOS branches
                if node.token_id == m_inf.eos_token_id and node is not root:
                    next_active.append(node)
                    continue
                # Reconstruct sequence and loop counts
                seq, loops = [], []
                n = node
                while n.prev_node:
                    seq.append(n.token_id)
                    loops.append(n.curr_loops)
                    n = n.prev_node
                seq.append(m_inf.bos_token_id)
                seq.reverse()
                # Determine loop count for this decode
                if previous_high_entropy and node.curr_loops < max_loops:
                    loops_to_request = node.curr_loops + 1
                else:
                    loops_to_request = 1
                # Decode logits with optional looping
                logits = m_inf.model.decoder(
                    tgt=torch.tensor([seq], device=m_inf.device),
                    memory=memory,
                    nmr_tokens=nmr_tokens_b,
                    num_loops=loops_to_request
                )[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = m_inf.calculate_entropy(log_probs)
                varentropy = m_inf.calculate_varentropy(log_probs)
                # Decide expansion strategy
                if entropy < entropy_threshold and varentropy < varentropy_threshold:
                    # Branch among top-k candidates
                    topk_vals, topk_ids = torch.topk(log_probs, top_k)
                    for lp, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
                        next_active.append(
                            EntropixNode(None, node, int(tid), node.log_prob + float(lp), node.length + 1, curr_loops=1)
                        )
                    previous_high_entropy = False
                elif entropy < entropy_threshold and varentropy >= varentropy_threshold:
                    # Greedy selection
                    best_id = int(log_probs.argmax().item())
                    next_active.append(
                        EntropixNode(None, node, best_id, node.log_prob + float(log_probs[best_id]), node.length + 1, curr_loops=1)
                    )
                    previous_high_entropy = False
                else:
                    # High entropy: stick with best after requested loops
                    best_id = int(log_probs.argmax().item())
                    next_active.append(
                        EntropixNode(None, node, best_id, node.log_prob + float(log_probs[best_id]), node.length + 1, curr_loops=loops_to_request)
                    )
                    previous_high_entropy = True
            # Prune to top_k
            active_nodes = sorted(next_active, reverse=True)[:top_k]
            # Build output list of (sequence IDs, loop_counts)
            output = []
            for node in active_nodes:
                seq_ids, loop_counts = [], []
                n = node
                while n.prev_node:
                    seq_ids.append(n.token_id)
                    loop_counts.append(n.curr_loops)
                    n = n.prev_node
                seq_ids.append(m_inf.bos_token_id)
                seq_ids.reverse(); loop_counts.reverse()
                output.append((seq_ids, loop_counts))
            yield output
            # Stop if all sequences have reached EOS
            if all(seq and seq[-1] == m_inf.eos_token_id for seq, _ in output):
                break 