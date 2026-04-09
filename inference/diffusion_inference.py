import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class DiffusionInference:
    """
    Iterative discrete denoising for the multimodal SMILES diffusion model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        ir_as_prompt: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.ir_as_prompt = ir_as_prompt

        self.bos_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must expose a [MASK] token for diffusion decoding.")

    def prepare_inputs(
        self,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if nmr_tokens is not None and nmr_tokens.dim() == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if ir_data is not None and ir_data.dim() == 1:
            ir_data = ir_data.unsqueeze(0)

        batch_size = 1
        if nmr_tokens is not None:
            batch_size = nmr_tokens.size(0)
        elif ir_data is not None:
            batch_size = ir_data.size(0)

        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(self.device)
        if ir_data is not None:
            ir_data = ir_data.to(self.device)
        return nmr_tokens, ir_data, batch_size

    def _decode_token_ids(self, sequences: torch.Tensor) -> List[str]:
        outputs: List[str] = []
        for seq in sequences.tolist():
            cleaned = []
            for token_id in seq:
                if token_id == self.eos_token_id:
                    break
                if token_id in (self.bos_token_id, self.pad_token_id, self.mask_token_id):
                    continue
                cleaned.append(token_id)
            outputs.append(self.tokenizer.decode(cleaned).replace(" ", "").strip())
        return outputs

    def _add_gumbel_noise(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0.0:
            return logits
        probs = logits.to(torch.float64).exp()
        noise = torch.rand_like(probs, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise.clamp_min(1e-12))).pow(temperature)
        return probs / gumbel_noise

    def _get_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer = torch.full(
            (mask_index.size(0), steps),
            0,
            dtype=torch.int64,
            device=mask_index.device,
        )
        if steps == 0:
            return num_transfer
        num_transfer += base
        for row in range(mask_index.size(0)):
            if remainder[row].item() > 0:
                num_transfer[row, : remainder[row].item()] += 1
        return num_transfer

    def _forward_logits(
        self,
        x: torch.Tensor,
        nmr_tokens: Optional[torch.Tensor],
        ir_data: Optional[torch.Tensor],
        cfg_scale: float,
        block_start: Optional[int] = None,
        block_end: Optional[int] = None,
    ) -> torch.Tensor:
        model_kwargs = dict(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            target_seq=x,
        )
        if block_start is not None and block_end is not None:
            model_kwargs["block_start"] = block_start
            model_kwargs["block_end"] = block_end
        logits = self.model(**model_kwargs)
        if cfg_scale > 0.0 and (nmr_tokens is not None or ir_data is not None):
            uncond_kwargs = dict(
                nmr_tokens=None,
                ir_data=None,
                target_seq=x,
            )
            if block_start is not None and block_end is not None:
                uncond_kwargs["block_start"] = block_start
                uncond_kwargs["block_end"] = block_end
            uncond_logits = self.model(**uncond_kwargs)
            logits = uncond_logits + (cfg_scale + 1.0) * (logits - uncond_logits)
        return logits

    @torch.no_grad()
    def decode_blockwise(
        self,
        nmr_tokens: Optional[torch.Tensor] = None,
        ir_data: Optional[torch.Tensor] = None,
        max_len: int = 128,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        cfg_scale: float = 0.0,
    ) -> List[str]:
        nmr_tokens, ir_data, batch_size = self.prepare_inputs(nmr_tokens, ir_data)
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to include BOS and one generated token.")

        gen_length = max_len - 1
        block_length = gen_length if block_length is None else block_length
        steps = max(1, block_length) if steps is None else steps
        if block_length <= 0:
            raise ValueError("block_length must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")

        x = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        x[:, 0] = self.bos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for block_start in range(1, max_len, block_length):
            block_end = min(block_start + block_length, max_len)
            active_rows = ~finished
            if not bool(active_rows.any()):
                break

            x[active_rows, block_start:block_end] = self.mask_token_id
            block_mask_index = x[:, block_start:block_end].eq(self.mask_token_id)
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps)

            for step_idx in range(steps):
                logits = self._forward_logits(
                    x=x,
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    cfg_scale=cfg_scale,
                    block_start=block_start,
                    block_end=block_end,
                )[:, block_start:block_end, :]
                logits[:, :, self.bos_token_id] = float("-inf")
                logits[:, :, self.pad_token_id] = float("-inf")
                logits[:, :, self.mask_token_id] = float("-inf")

                logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                mask_index = x[:, block_start:block_end].eq(self.mask_token_id)
                current_block = x[:, block_start:block_end]
                x0 = torch.where(mask_index, x0, current_block)

                if remasking == "low_confidence":
                    probs = F.softmax(logits.float(), dim=-1)
                    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    confidence = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unsupported remasking strategy: {remasking}")

                confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))

                transfer_index = torch.zeros_like(mask_index)
                for row in range(batch_size):
                    if finished[row]:
                        continue
                    k = int(num_transfer_tokens[row, step_idx].item())
                    if k <= 0:
                        continue
                    _, selected = torch.topk(confidence[row], k=k)
                    transfer_index[row, selected] = True
                current_block = torch.where(transfer_index, x0, current_block)
                x[:, block_start:block_end] = current_block

            for row in range(batch_size):
                if finished[row]:
                    continue
                block_tokens = x[row, block_start:block_end]
                eos_hits = torch.nonzero(block_tokens.eq(self.eos_token_id), as_tuple=False)
                if eos_hits.numel() == 0:
                    continue
                eos_pos = int(eos_hits[0].item())
                if eos_pos + 1 < block_tokens.numel():
                    x[row, block_start + eos_pos + 1 : block_end] = self.pad_token_id
                if block_end < max_len:
                    x[row, block_end:] = self.pad_token_id
                finished[row] = True

        return self._decode_token_ids(x[:, 1:])

    @torch.no_grad()
    def decode(
        self,
        nmr_tokens: Optional[torch.Tensor] = None,
        ir_data: Optional[torch.Tensor] = None,
        max_len: int = 128,
        steps: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        cfg_scale: float = 0.0,
    ) -> List[str]:
        if getattr(self.model, "supports_block_diffusion", False):
            return self.decode_blockwise(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                max_len=max_len,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                cfg_scale=cfg_scale,
            )

        nmr_tokens, ir_data, batch_size = self.prepare_inputs(nmr_tokens, ir_data)
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to include BOS and one generated token.")

        gen_length = max_len - 1
        block_length = gen_length if block_length is None else block_length
        steps = gen_length if steps is None else steps
        if block_length <= 0:
            raise ValueError("block_length must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")

        num_blocks = math.ceil(gen_length / block_length)
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of generation blocks.")
        steps_per_block = steps // num_blocks

        x = torch.full((batch_size, max_len), self.mask_token_id, dtype=torch.long, device=self.device)
        x[:, 0] = self.bos_token_id

        for block_idx in range(num_blocks):
            block_start = 1 + block_idx * block_length
            block_end = min(block_start + block_length, max_len)
            block_mask_index = x[:, block_start:block_end].eq(self.mask_token_id)
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step_idx in range(steps_per_block):
                mask_index = x.eq(self.mask_token_id)
                logits = self._forward_logits(
                    x=x,
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    cfg_scale=cfg_scale,
                )
                logits[:, :, self.bos_token_id] = float("-inf")
                logits[:, :, self.pad_token_id] = float("-inf")
                logits[:, :, self.mask_token_id] = float("-inf")

                logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0 = torch.where(mask_index, x0, x)

                if remasking == "low_confidence":
                    probs = F.softmax(logits.float(), dim=-1)
                    confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    confidence = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unsupported remasking strategy: {remasking}")

                confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
                confidence[:, block_end:] = float("-inf")

                transfer_index = torch.zeros_like(mask_index)
                for row in range(batch_size):
                    k = int(num_transfer_tokens[row, step_idx].item())
                    if k <= 0:
                        continue
                    _, selected = torch.topk(confidence[row], k=k)
                    transfer_index[row, selected] = True
                x[transfer_index] = x0[transfer_index]

        return self._decode_token_ids(x[:, 1:])
