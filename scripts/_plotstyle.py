"""
Shared cosmetic constants for the plotting scripts.

Only constants that are byte-identical across multiple plotters live here. Each
plotter keeps any palette/label set unique to it (e.g. PTQ's two-colour
model_a/model_b scheme, Merlin's 9-config palette, the per-layer CKA colours).
"""

# Vision quantization configs (least → most aggressive)
CONFIG_LABELS = {
    "fp":   "FP",
    "w8a8": "W8A8",
    "w6a6": "W6A6",
    "w4a6": "W4A6",
    "w4a4": "W4A4",
    "w2a4": "W2A4",
}
CONFIG_ORDER = list(CONFIG_LABELS.keys())

# Teacher vs. student grouped-bar palette
TEACHER_COLOR = "#6B7280"
STUDENT_COLOR = "#2563EB"
DELTA_POS     = "#16A34A"
DELTA_NEG     = "#DC2626"

# Display-name maps
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "stl10":   "STL-10",
}
BACKBONE_LABELS = {
    "resnet18":           "ResNet-18",
    "efficientnet_b0":    "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNet-V3-Small",
    "vit_b_16":           "ViT-B/16",
    "swin_t":             "Swin-T",
}

# Multi-run line/star colour cycle (QAT comparison plots)
QAT_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C"]
