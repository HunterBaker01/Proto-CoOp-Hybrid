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

    def __init__(
        self, clip_model, classnames, n_ctx, ctx_init, class_token_position, csc=False
    ):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Initialize context vectors. If ctx_init is provided, use its embeddings.
        # Otherwise, initialize with random noise (X's).
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(
                clip_model.token_embedding.weight.device
            )
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f"Initial context: '{prompt_prefix}'")
        print(f"Number of context words (tokens): {n_ctx}")

        # The learnable context vectors are registered as a parameter.
        self.ctx = nn.Parameter(ctx_vectors)

        # Tokenize all prompts (e.g., "X X X X classname.")
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            clip_model.token_embedding.weight.device
        )

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # Register the start and end token embeddings as non-learnable buffers.
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS token
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx :, :]
        )  # Class and EOS tokens

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = class_token_position

    def forward(self):
        """Constructs the full prompt embeddings by combining context and class tokens."""
        ctx = self.ctx
        # Expand context to match the number of classes
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # Combine the start token, context, and class+end tokens
        prompts = torch.cat(
            [
                self.token_prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                self.token_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class OurCLIP(nn.Module):
    """The main model class that integrates the PromptLearner with CLIP."""

    def __init__(self, classnames, n_ctx, ctx_init, class_token_position, csc=False):
        super().__init__()
        clip_model, preprocess = clip.load("ViT-B/16")
        clip_model = clip_model.float()

        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx, ctx_init, class_token_position, csc=csc
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        image_features = self.image_encoder(image)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # Normalize features to compute cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
