from data_processing import CLASS_NAMES
from model.clip_coop_wrapper import OurCLIP
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def train_coop(
    fewshot_loader,
    class_indices,
    num_epochs=5,
    lr=1e-3,
    n_ctx=12,
    ctx_init=None,
    class_token_position="end",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train CoOp's prompt vectors on base classes.
    """
    print(f"Training CoOp on {len(class_indices)} base classes")
    classnames = [CLASS_NAMES[i] for i in class_indices]
    coop_model = OurCLIP(classnames, n_ctx, ctx_init, class_token_position).to(device)

    # Freeze all model parameters except for the context vectors in the prompt learner
    for param in coop_model.parameters():
        param.requires_grad = False
    coop_model.prompt_learner.ctx.requires_grad = True

    optimizer = optim.Adam([coop_model.prompt_learner.ctx], lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training the model (epochs passed in the function)
    for epoch in range(num_epochs):
        coop_model.train()
        epoch_loss = 0.0

        # Standard loop for training
        for images, labels in tqdm(
            fewshot_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = coop_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(fewshot_loader)
        print(
            f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f}"
        )  # Show the loss for that epoch

    # We want to extract the learned context prompt so we clone the original to avoid
    # any overwrite issues and gradient tracking
    coop_model.eval()
    learned_ctx = coop_model.prompt_learner.ctx.clone()

    # Keep the training config just to keep a log
    training_config = {
        "n_ctx": n_ctx,
        "ctx_init": ctx_init,
        "class_token_position": class_token_position,
    }

    return coop_model, learned_ctx, training_config
