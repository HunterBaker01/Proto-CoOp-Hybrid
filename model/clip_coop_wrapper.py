import torch
import clip
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    A custiom encoder that wraps the CLIP transformer.
    Necessary to use the CLIP model with the COOP framework.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding

        # Permute the dimensions to match the expected input shape: [n_ctx, batch_size, transformer.width]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        # Permute back to the original shape: [batch_size, n_ctx, transformer.width]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # Extract the feature corresponding to the [EOT] token for each prompt
        # and project it to the final embedding space
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x


class PromptLearner(nn.Module):
    """
    Implements the core functionality of the COOP framework.
    """
