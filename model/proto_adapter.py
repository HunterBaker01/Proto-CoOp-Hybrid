import torch
import tqdm


def build_prototypes(train_loader, class_indices, device="cuda", model=None):
    """
    Build visual prototypes by averaging image features from the training data for each class.
    (Proto-adapter)
    """
    num_classes = len(class_indices)
    features_per_class = {i: [] for i in range(num_classes)}

    # Map original class labels (e.g., 0-101) to new indices (e.g., 0-50)
    label_to_idx = {
        orig_label: new_idx for new_idx, orig_label in enumerate(class_indices)
    }

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building prototypes"):
            images = images.to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.float()

            for feat, label in zip(features, labels):
                if label.item() in label_to_idx:
                    new_idx = label_to_idx[label.item()]
                    features_per_class[new_idx].append(feat.cpu())

    # Compute the mean feature for each class to create the prototype
    prototypes = []
    for i in range(num_classes):
        if len(features_per_class[i]) > 0:
            feats = torch.stack(features_per_class[i])
            proto = feats.mean(dim=0)
            proto = proto / proto.norm()  # Normalize the final prototype
            prototypes.append(proto)
        else:
            # If there are no training samples in the class just fill with zeros
            prototypes.append(torch.zeros(512))

    return torch.stack(prototypes).to(device)
