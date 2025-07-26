import torchvision
from torch.utils.data import Subset

# The following code defines the class names for a dataset of flowers. Each class name corresponds to a specific type of flower.
CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def get_data(data_dir="./data", transform=None):
    """Load the Flowers102 train, val, and test sets."""
    train = torchvision.datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=transform
    )
    val = torchvision.datasets.Flowers102(
        root=data_dir, split="val", download=True, transform=transform
    )
    test = torchvision.datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=transform
    )
    return train, val, test


def base_novel_categories():
    """Return the base and novel categories for the Flowers102 dataset."""
    num_classes = len(CLASS_NAMES)
    base_classes = list(range(num_classes // 2))
    novel_classes = list(range(num_classes // 2, num_classes))
    return base_classes, novel_classes


def split_data(dataset, base_classes):
    """Split the dataset into base and novel categories."""
    base_categories_samples = []
    novel_categories_samples = []
    base_set = set(base_classes)

    for sample_id, label in enumerate(dataset.targets):
        if label in base_set:
            base_categories_samples.append(sample_id)
        else:
            novel_categories_samples.append(sample_id)

    base_dataset = Subset(dataset, base_categories_samples)
    novel_dataset = Subset(dataset, novel_categories_samples)
    return base_dataset, novel_dataset
