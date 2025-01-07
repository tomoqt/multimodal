import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HigherOrderMultiInputCrossAttention(nn.Module):
    """
    A higher-order cross-attention mechanism that generalizes multi-modal attention
    to arbitrary order >= 2, *without* causal masks or top-k pruning.

    - We assume you have 'order' input modalities (e.g., IR, 1H NMR, HSQC, etc.),
      each shaped (B, T_i, C).
    - For order=2, it reduces to a standard cross-attention across two sequences.
    - For order > 2, we build a multi-dimensional attention tensor A via einsum,
      then contract it with the corresponding V-factors.

    Usage:
      1) Initialize with config specifying n_head, n_embd, order, dropout, bias.
      2) Forward takes a list of `order` Tensors as input, each shape (B, T_i, C).
      3) Returns a single output Tensor shape (B, T_1, C), i.e. using the
         first input's length as the "query" dimension.

    Note:
      - By default, we treat the *first* sequence as the "query dimension"
        and all sequences (including the first) as "key" dimensions for the
        attention. That's one possible design choice; you can adapt as needed.
      - No masking is applied; all positions can attend to all positions in
        each modality. If you need masking, you can add it manually.
    """

    def __init__(self, config):
        """
        config must have:
          - n_head: number of attention heads (int)
          - n_embd: total embedding dimension (int), must be divisible by n_head
          - order: how many input sequences/modalities
          - dropout: dropout probability (float)
          - bias: bool, whether linear layers use bias
        Example:
          config.n_head = 8
          config.n_embd = 256
          config.order = 3
          config.dropout = 0.1
          config.bias = True
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // self.n_head
        self.dropout = config.dropout
        self.order = config.order
        assert self.order >= 2, "Minimum attention order is 2."

        # For indexing in einsum patterns (up to 10D)
        self.index_letters = ['i','j','k','l','m','n','o','p','q','r']
        assert self.order <= len(self.index_letters), "Order too large for index letters."

        # For each head, we create:
        #   Q_factors: order Linear layers => one per input
        #   K_factors: order Linear layers => one per input
        #   V_factors: order-1 Linear layers => one for each input except the first?
        #   (Design choice: the first sequence is the "query dimension," so
        #    all sequences are "keys." We still produce K_i, V_i for each modality
        #    except we omit the final V for the first dimension if you want an
        #    output shaped by the first dimension. However, you might prefer
        #    that each input has V. There's no single "correct" approach.)
        #
        #   In this example, we'll let each input have Q_i, K_i, and for values
        #   we'll have V_i for i in [1..order-1]. The first input is "queries",
        #   the rest are "keys/values." If you need a fully symmetric approach,
        #   adapt accordingly.
        #
        self.q_projections = nn.ModuleList()  # each entry: list[Linear(...), ...]
        self.k_projections = nn.ModuleList()
        self.v_projections = nn.ModuleList()

        for _ in range(self.n_head):
            q_list = nn.ModuleList([
                nn.Linear(config.n_embd, self.head_size, bias=config.bias)
                for _ in range(self.order)
            ])
            k_list = nn.ModuleList([
                nn.Linear(config.n_embd, self.head_size, bias=config.bias)
                for _ in range(self.order)
            ])
            # For simplicity, let's give *every* input a V factor. Then after
            # building multi-dimensional A, we can contract with each input's V
            # but often you want the final shape to be (B, T_first, head_size).
            v_list = nn.ModuleList([
                nn.Linear(config.n_embd, self.head_size, bias=config.bias)
                for _ in range(self.order)
            ])
            self.q_projections.append(q_list)
            self.k_projections.append(k_list)
            self.v_projections.append(v_list)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, *inputs):
        """
        Forward pass for higher-order cross-attention.

        inputs: a variable number of Tensors (self.order many),
                each shape (B, T_i, C).
        We'll interpret the 1st input as "query dimension" (T_1),
        and all inputs as "key dimensions" in the attention.

        Returns: (B, T_1, C) - an output shaped along the first input's sequence length.
        """
        assert len(inputs) == self.order, f"Expected {self.order} input tensors."
        B = inputs[0].shape[0]

        # We'll accumulate heads in a list
        heads_output = []

        for head_idx in range(self.n_head):
            # For this head, gather Q_factors, K_factors, V_factors
            Q_factors = []
            K_factors = []
            V_factors = []

            for i in range(self.order):
                q_proj = self.q_projections[head_idx][i]
                k_proj = self.k_projections[head_idx][i]
                v_proj = self.v_projections[head_idx][i]

                # shape => (B, T_i, head_size)
                Q_factors.append(q_proj(inputs[i]))
                K_factors.append(k_proj(inputs[i]))
                V_factors.append(v_proj(inputs[i]))

            # (Optional) We can do a multi-dimensional attention with:
            #   A = einsum( Q0, K1, K2, ... ) => shape (B, T0, T1, T2, ...)
            # Then softmax across all "non-query" dims. Finally multiply by V1, V2, etc.
            # Here we do a simpler approach: order=2 => standard cross-attn,
            #   else do a straightforward N-D approach.

            if self.order == 2:
                # Standard cross-attention
                Q = Q_factors[0]    # (B, T0, hsize)
                K = K_factors[1]    # (B, T1, hsize)
                V = V_factors[1]    # (B, T1, hsize)

                # (B,T0,T1)
                A = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
                A = F.softmax(A, dim=-1)
                A = self.attn_dropout(A)
                Y = A @ V   # (B,T0,hsize)

            else:
                # Higher-order: e.g. order=3 => Q_factors[0], K_factors[1], K_factors[2].
                # We'll define that the first factor is Q (B,T0,hsize), and the rest are K.
                # Then we multiply them all via einsum => shape (B, T0, T1, T2, ...).
                # For example, order=3 => dims = [i,j,k], Q0 => i, K1 => j, K2 => k, etc.

                dims = self.index_letters[:self.order]
                # factor_list[0] => Q_factors[0], dimension i
                # factor_list[1..] => K_factors[1..], dimension j,k,...
                factor_list = []
                factor_list.append(Q_factors[0])  # (B, T0, hsize)
                for iK in range(1, self.order):
                    factor_list.append(K_factors[iK])  # (B, T_i, hsize)

                # Build the einsum pattern:
                # e.g. for order=3 => "b i h, b j h, b k h -> b i j k"
                in_subscripts = []
                for dim_idx, dim_letter in enumerate(dims):
                    in_subscripts.append(f"b{dim_letter}h")
                out_subscript = "b" + "".join(dims)  # e.g. "bijk"
                einsum_str_A = f"{','.join(in_subscripts)}->{out_subscript}"

                A = torch.einsum(einsum_str_A, *factor_list)
                A = A / math.sqrt(self.head_size)

                # Softmax over *all but the first* dimension => flatten them
                # shape(A) => (B, T0, T1, T2, ...)
                A_shape = A.shape
                T0 = A_shape[1]   # first dimension
                other_dims_size = 1
                for d in A_shape[2:]:
                    other_dims_size *= d
                A = A.view(B, T0, other_dims_size)
                A = F.softmax(A, dim=-1)
                A = self.attn_dropout(A)
                A = A.view(*A_shape)  # restore shape

                # Next, we contract A with the V factors. We'll treat
                # all V_factors except V_factors[0] as "keys" to combine
                # along the same dims. If you want to also include V_factors[0],
                # you'd do a symmetrical approach.
                #
                # For example, order=3 => we have V1 => (B, T1, hsize), V2 => (B, T2, hsize)
                # We'll do an outer product for V1, V2 => shape (B, T1, T2, hsize)
                # then multiply with A => shape (B, T0, hsize)
                # This is just one design approach. Adjust to your needs.

                # Let's merge all V_factors[1..] by an outer product:
                # step by step
                merged_V = V_factors[1]  # (B, T1, hsize)
                merged_dims = [dims[1]]  # e.g. ['j']

                for iV in range(2, self.order):
                    merged_V = self._outer_product_expand(merged_V, merged_dims,
                                                          V_factors[iV], dims[iV])
                    merged_dims.append(dims[iV])

                # Now merged_V => shape (B, T1, T2, ..., hsize)
                # A => shape (B, T0, T1, T2, ...)
                # final einsum => "b i j k, b j k h -> b i h" for order=3
                # general => in_sub_A="b i j k...", in_sub_V="b j k... h", out_sub="b i h"
                in_sub_A = "b" + "".join(dims)
                in_sub_V = "b" + "".join(dims[1:]) + "h"
                out_sub = f"b{dims[0]}h"
                einsum_str_final = f"{in_sub_A},{in_sub_V}->{out_sub}"

                Y = torch.einsum(einsum_str_final, A, merged_V)  # (B, T0, hsize)

            heads_output.append(Y)

        # Combine heads: (B, T0, n_head, head_size)
        out = torch.stack(heads_output, dim=2)  # (B, T0, n_head, head_size)
        out = out.reshape(B, inputs[0].shape[1], self.n_embd)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out

    def _outer_product_expand(self, tensorA, dimsA, tensorB, dimB):
        """
        Merges 'tensorA' with shape (B, T_{dimsA[0]}, T_{dimsA[1]},..., h)
        and 'tensorB' with shape (B, T_{dimB}, h) via an outer product across 'h'.

        Example (order=3):
          tensorA: (B, T_j, h)
          tensorB: (B, T_k, h)
          dimsA=['j'], dimB='k'
          => "b j h, b k h -> b j k h"
        """
        in_sub_A = f"b{''.join(dimsA)}h"
        in_sub_B = f"b{dimB}h"
        out_sub = f"b{''.join(dimsA)}{dimB}h"
        einsum_str = f"{in_sub_A},{in_sub_B}->{out_sub}"
        return torch.einsum(einsum_str, tensorA, tensorB)
