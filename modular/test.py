import torch
import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from data_processing import get_data, base_novel_categories, split_data
from model.clip_coop_wrapper import OurCLIP

clip_wrapper = OurCLIP()
model = clip_wrapper.model
preprocess = clip_wrapper.preprocess

def evaluate(
    test_loader,
    class_indices,
    text_features,
    proto_weights=None,
    alpha=0.55,
    conf_threshold=0.98,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model=model,
):
    """Evaluate the model on test data using the confidence-based proto-adapter switching mechanism."""
    all_preds = []
    all_labels = []
    all_confidences = []

    label_to_idx = {
        orig_label: new_idx for new_idx, orig_label in enumerate(class_indices)
    }

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.float()

            clip_logits = 100.0 * features @ text_features.T

            if proto_weights is not None and alpha > 0.0:
                proto_logits = 100.0 * features @ proto_weights.T
                proto_probs = F.softmax(proto_logits, dim=-1)
                max_conf, _ = proto_probs.max(dim=-1)
                all_confidences.extend(max_conf.cpu().numpy())

                combined_logits = []
                for i in range(features.shape[0]):
                    if max_conf[i] > conf_threshold:
                        mixed_logits = (
                            alpha * proto_logits[i]
                            + (1 - alpha) * clip_logits[i, : proto_logits.shape[1]]
                        )
                        if proto_logits.shape[1] < clip_logits.shape[1]:
                            mixed_logits = F.pad(
                                mixed_logits,
                                (0, clip_logits.shape[1] - proto_logits.shape[1]),
                                value=float("-inf"),
                            )
                        combined_logits.append(mixed_logits)
                    else:
                        combined_logits.append(clip_logits[i])
                combined_logits = torch.stack(combined_logits)
            else:
                all_confidences.extend(np.zeros(images.shape[0]))
                combined_logits = clip_logits

            preds = combined_logits.argmax(dim=-1).cpu()
            mapped_labels = [label_to_idx.get(label.item(), -1) for label in labels]
            all_preds.extend(preds.numpy())
            all_labels.extend(mapped_labels)

    valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]

    acc = accuracy_score(filtered_labels, filtered_preds) if filtered_labels else 0.0
    return acc, np.array(all_confidences)


def single_experiment(ctx, alpha, conf_threshold, lr, epochs, batch_size=16):
    """Run a single, full experiment with a given set of parameters."""
    print(f"\n{'=' * 60}")
    print(f"Running experiment with:")
    print(
        f"CTX={ctx}, ALPHA={alpha}, CONF_THRESHOLD={conf_threshold}, LR={lr}, EPOCHS={epochs}"
    )
    print(f"{'=' * 60}")

    train_set, _, test_set = get_data(transform=preprocess)
    base_classes, novel_classes = base_novel_categories()
    train_base, _ = split_data(train_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)
    train_loader_base = DataLoader(train_base, batch_size=batch_size, shuffle=True)
    test_loader_base = DataLoader(test_base, batch_size=batch_size, shuffle=False)
    test_loader_novel = DataLoader(test_novel, batch_size=batch_size, shuffle=False)

    _, learned_ctx, training_config = train_coop(
        train_loader_base, base_classes, num_epochs=epochs, lr=lr, n_ctx=ctx
    )

    base_text_features = get_text_features_with_learned_prompt(
        base_classes, learned_ctx, training_config
    )
    novel_text_features = get_text_features_with_learned_prompt(
        novel_classes, learned_ctx, training_config
    )

    base_proto_weights = build_prototypes(train_loader_base, base_classes)

    base_acc, base_confidences = evaluate(
        test_loader_base,
        base_classes,
        base_text_features,
        base_proto_weights,
        alpha=alpha,
        conf_threshold=conf_threshold,
    )
    novel_acc, novel_confidences = evaluate(
        test_loader_novel,
        novel_classes,
        novel_text_features,
        base_proto_weights,
        alpha=alpha,
        conf_threshold=conf_threshold,
    )

    return base_acc, novel_acc


def run_experiments():
    """Define and run a set of experiments, then display the results."""
    # Running only the best configuration from the report for this demonstration
    experiment_configs = [
        {"ctx": 13, "alpha": 0.55, "conf_threshold": 0.985, "lr": 0.001, "epochs": 5},
    ]

    results = []
    for i, config in enumerate(experiment_configs):
        print(f"\nExperiment {i + 1}/{len(experiment_configs)}")
        try:
            base_acc, novel_acc = single_experiment(**config)
            result = {
                "experiment_id": i + 1,
                **config,
                "base_accuracy": base_acc * 100,
                "novel_accuracy": novel_acc * 100,
                "status": "success",
            }
            print(
                f"✅ Experiment {i + 1} completed: Base Acc: {base_acc * 100:.2f}%, Novel Acc: {novel_acc * 100:.2f}%"
            )
        except Exception as e:
            print(f"❌ Experiment {i + 1} failed: {str(e)}")
            result = {
                "experiment_id": i + 1,
                **config,
                "base_accuracy": None,
                "novel_accuracy": None,
                "status": f"failed: {str(e)}",
            }
        results.append(result)

    df = pd.DataFrame(results)
    print("\n📊 RESULTS SUMMARY:")
    print("=" * 100)
    print(
        df[
            [
                "experiment_id",
                "ctx",
                "alpha",
                "conf_threshold",
                "lr",
                "epochs",
                "base_accuracy",
                "novel_accuracy",
            ]
        ].to_string(index=False)
    )
    return df
