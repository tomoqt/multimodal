import enum
from typing import List, Optional, Tuple

import torch


class DecodingStrategy(enum.Enum):
    GREEDY = "greedy"
    BEAM = "beam"
    SAMPLING = "sampling"
    NUCLEUS = "nucleus"


class ModelInference:
    """
    Lean inference wrapper around the core model.
    """

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None, ir_as_prompt: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.ir_as_prompt = ir_as_prompt

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
        nmr_padding_mask: Optional[torch.Tensor] = None,
        **_: dict,
    ) -> List[str]:
        del mass_data
        if strategy == DecodingStrategy.GREEDY:
            return self.greedy_decode(nmr_tokens, ir_data, max_len, nmr_padding_mask=nmr_padding_mask)
        if strategy == DecodingStrategy.BEAM:
            return self.beam_search(
                nmr_tokens,
                ir_data,
                max_len,
                beam_width,
                length_penalty,
                nmr_padding_mask=nmr_padding_mask,
            )
        if strategy in (DecodingStrategy.SAMPLING, DecodingStrategy.NUCLEUS):
            return self.sample_decode(
                nmr_tokens,
                ir_data,
                max_len,
                temperature,
                top_k,
                top_p,
                nmr_padding_mask=nmr_padding_mask,
            )
        raise ValueError(f"Unsupported strategy: {strategy}")

    def prepare_inputs(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
    ) -> Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], int]:
        if nmr_tokens is not None and nmr_tokens.dim() == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if ir_data is not None and ir_data.dim() == 1:
            ir_data = ir_data.unsqueeze(0)

        batch_size = 1
        if nmr_tokens is not None:
            batch_size = nmr_tokens.size(0)
        elif ir_data is not None:
            batch_size = ir_data.size(0)

        return (nmr_tokens, ir_data), batch_size

    def encode_inputs(self, nmr_tokens: Optional[torch.Tensor], ir_data: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = 1
        if nmr_tokens is not None:
            batch_size = nmr_tokens.size(0)
        elif ir_data is not None:
            batch_size = ir_data.size(0)

        if self.ir_as_prompt:
            if ir_data is None:
                return torch.zeros(batch_size, 1, self.model.embed_dim, device=self.device)
            return self.model.ir_embed(ir_data.to(self.device).long())

        if ir_data is None:
            return torch.zeros(batch_size, self.model.max_memory_length, self.model.embed_dim, device=self.device)
        return self.model.encoder(None, ir_data.to(self.device), None)

    def _decode_token_ids(self, sequences: List[List[int]]) -> List[str]:
        outputs: List[str] = []
        for seq in sequences:
            cleaned: List[int] = []
            for token_id in seq:
                if token_id == self.eos_token_id:
                    break
                if token_id in (self.bos_token_id, self.pad_token_id):
                    continue
                cleaned.append(token_id)
            outputs.append(self.tokenizer.decode(cleaned).replace(" ", "").strip())
        return outputs

    def greedy_decode(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
        max_len: int,
        nmr_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        (nmr_tokens, ir_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data)
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(self.device)
        if ir_data is not None:
            ir_data = ir_data.to(self.device)
        if nmr_padding_mask is not None:
            nmr_padding_mask = nmr_padding_mask.to(self.device)

        memory = self.encode_inputs(nmr_tokens, ir_data)
        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            logits = self.model.decoder(seq, memory, nmr_tokens=nmr_tokens, nmr_padding_mask=nmr_padding_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token_id)
            if bool(finished.all()):
                break

        return self._decode_token_ids(seq.tolist())

    def _top_k_top_p_filter(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        filtered = logits.clone()
        if top_k > 0:
            kth_vals = torch.topk(filtered, k=min(top_k, filtered.size(-1)), dim=-1).values[:, -1].unsqueeze(-1)
            filtered = torch.where(filtered < kth_vals, torch.full_like(filtered, float("-inf")), filtered)

        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            remove = cumprobs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(filtered, float("-inf"))
            filtered.scatter_(1, sorted_idx, sorted_logits)
        return filtered

    def sample_decode(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
        max_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
        nmr_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        (nmr_tokens, ir_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data)
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(self.device)
        if ir_data is not None:
            ir_data = ir_data.to(self.device)
        if nmr_padding_mask is not None:
            nmr_padding_mask = nmr_padding_mask.to(self.device)

        memory = self.encode_inputs(nmr_tokens, ir_data)
        seq = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        temperature = max(temperature, 1e-4)
        for _ in range(max_len):
            logits = self.model.decoder(
                seq,
                memory,
                nmr_tokens=nmr_tokens,
                nmr_padding_mask=nmr_padding_mask,
            )[:, -1, :] / temperature
            logits = self._top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token_id)
            if bool(finished.all()):
                break

        return self._decode_token_ids(seq.tolist())

    def beam_search_nbest(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
        max_len: int,
        beam_width: int,
        length_penalty: float,
        n_best: Optional[int] = None,
        nmr_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[str]]:
        (nmr_tokens, ir_data), batch_size = self.prepare_inputs(nmr_tokens, ir_data)
        beam_width = max(1, int(beam_width))
        n_best = beam_width if n_best is None else max(1, min(int(n_best), beam_width))

        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(self.device)
        if ir_data is not None:
            ir_data = ir_data.to(self.device)
        if nmr_padding_mask is not None:
            nmr_padding_mask = nmr_padding_mask.to(self.device)

        memory = self.encode_inputs(nmr_tokens, ir_data)
        memory = memory.repeat_interleave(beam_width, dim=0)
        nmr_tokens = nmr_tokens.repeat_interleave(beam_width, dim=0) if nmr_tokens is not None else None
        nmr_padding_mask = (
            nmr_padding_mask.repeat_interleave(beam_width, dim=0) if nmr_padding_mask is not None else None
        )

        seq = torch.full(
            (batch_size * beam_width, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        beam_scores = torch.full((batch_size, beam_width), float("-inf"), device=self.device)
        beam_scores[:, 0] = 0.0
        beam_finished = torch.zeros((batch_size, beam_width), dtype=torch.bool, device=self.device)
        beam_lengths = torch.ones((batch_size, beam_width), dtype=torch.long, device=self.device)
        vocab_size = None
        eos_only = None

        for _ in range(max_len):
            logits = self.model.decoder(
                seq,
                memory,
                nmr_tokens=nmr_tokens,
                nmr_padding_mask=nmr_padding_mask,
            )[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.view(batch_size, beam_width, vocab_size)

            if beam_finished.any():
                if eos_only is None or eos_only.size(-1) != vocab_size:
                    eos_only = torch.full((1, 1, vocab_size), float("-inf"), device=self.device)
                    eos_only[..., self.eos_token_id] = 0.0
                log_probs = torch.where(beam_finished.unsqueeze(-1), eos_only, log_probs)

            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            current_lengths = beam_lengths.unsqueeze(-1).expand(-1, -1, vocab_size)
            unfinished = (~beam_finished).unsqueeze(-1)
            candidate_lengths = current_lengths + unfinished.to(dtype=current_lengths.dtype)
            if length_penalty > 0:
                rank_scores = candidate_scores / candidate_lengths.to(dtype=torch.float32).pow(length_penalty)
            else:
                rank_scores = candidate_scores

            flat_rank = rank_scores.view(batch_size, beam_width * vocab_size)
            top_rank, top_pos = torch.topk(flat_rank, k=beam_width, dim=-1)
            del top_rank
            flat_scores = candidate_scores.view(batch_size, beam_width * vocab_size)
            next_beam_scores = torch.gather(flat_scores, 1, top_pos)
            parent_beam = torch.div(top_pos, vocab_size, rounding_mode="floor")
            next_token = top_pos % vocab_size

            base = (torch.arange(batch_size, device=self.device).unsqueeze(1) * beam_width)
            gather_rows = (base + parent_beam).reshape(-1)
            seq = seq[gather_rows]
            memory = memory[gather_rows]
            if nmr_tokens is not None:
                nmr_tokens = nmr_tokens[gather_rows]
            if nmr_padding_mask is not None:
                nmr_padding_mask = nmr_padding_mask[gather_rows]

            seq = torch.cat([seq, next_token.reshape(-1, 1)], dim=1)
            beam_scores = next_beam_scores
            parent_finished = torch.gather(beam_finished, 1, parent_beam)
            beam_finished = parent_finished | next_token.eq(self.eos_token_id)
            parent_lengths = torch.gather(beam_lengths, 1, parent_beam)
            beam_lengths = parent_lengths + (~parent_finished).to(dtype=parent_lengths.dtype)

            if bool(beam_finished.all()):
                break

        if length_penalty > 0:
            final_rank = beam_scores / beam_lengths.to(dtype=torch.float32).pow(length_penalty)
        else:
            final_rank = beam_scores
        order = torch.argsort(final_rank, dim=1, descending=True)
        seq = seq.view(batch_size, beam_width, -1)

        batch_outputs: List[List[str]] = []
        for batch_idx in range(batch_size):
            seen = set()
            decoded_rows: List[str] = []
            for beam_idx in order[batch_idx].tolist():
                decoded = self._decode_token_ids([seq[batch_idx, beam_idx].tolist()])[0]
                if decoded in seen:
                    continue
                seen.add(decoded)
                decoded_rows.append(decoded)
                if len(decoded_rows) >= n_best:
                    break
            batch_outputs.append(decoded_rows)
        return batch_outputs

    def beam_search(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
        max_len: int,
        beam_width: int,
        length_penalty: float,
        nmr_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        nbest = self.beam_search_nbest(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            max_len=max_len,
            beam_width=beam_width,
            length_penalty=length_penalty,
            n_best=1,
            nmr_padding_mask=nmr_padding_mask,
        )
        return [rows[0] if rows else "" for rows in nbest]
