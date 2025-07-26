import torch
from torch.utils.data import DataLoader
from get_data import CLASS_NAMES
import clip
from tqdm import tqdm


@torch.no_grad()
def eval_zeroshot(model, dataset, categories, batch_size=16, device="cuda", label=""):
    model.eval()
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}." for c in categories]
    ).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=f"Evaluating {label}"):
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()
        image = image.to(device)
        target = target.to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        predicted_classes = (image_features @ text_features.T).argmax(dim=-1)
        correct_predictions += (predicted_classes == target).sum().item()
    accuracy = correct_predictions / len(dataset)
    return accuracy
