"""
Jednorazowy export modelu PyTorch → ONNX
=========================================
Uruchom raz:  python export_to_onnx.py
Wynik:        bestMN.onnx  (obok tego skryptu)
"""

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from pathlib import Path

_DIR      = Path(__file__).parent
PT_FILE   = _DIR / "bestMN.pt"
ONNX_FILE = _DIR / "bestMN.onnx"
NUM_CLASSES = 3   # __background__, Person, Human head

print("[1/4] Ładowanie modelu PyTorch…")
device = torch.device("cpu")  # export zawsze na CPU

model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=None,
    num_classes=NUM_CLASSES,
)
state = torch.load(PT_FILE, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()
print("    OK")

# ── Wrapper który sprawia że ONNX export działa z SSD ──────────────────
# torchvision SSD ma skomplikowane wyjście (lista słowników) których
# ONNX nie lubi. Wrapper spłaszcza je do 3 tensorów.
class SSDWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        # x: tensor (1, 3, 320, 320)
        outs = self.m([x[0]])[0]         # lista → pierwszy obraz
        return outs["boxes"], outs["scores"], outs["labels"].float()

wrapped = SSDWrapper(model)
wrapped.eval()

print("[2/4] Tworzenie dummy input 320×320…")
dummy = torch.zeros(1, 3, 320, 320)    # jednorazowy "testowy" obraz
print("    OK")

print("[3/4] Export do ONNX…")
torch.onnx.export(
    wrapped,
    dummy,
    str(ONNX_FILE),
    opset_version    = 11,
    input_names      = ["image"],
    output_names     = ["boxes", "scores", "labels"],
    dynamic_axes     = {
        "image":  {0: "batch"},
        "boxes":  {0: "num_detections"},
        "scores": {0: "num_detections"},
        "labels": {0: "num_detections"},
    },
)
print(f"    Zapisano: {ONNX_FILE}")

print("[4/4] Weryfikacja pliku ONNX…")
import onnx
onnx.checker.check_model(str(ONNX_FILE))
print("    Model poprawny!")
print()
print(f"Gotowe! Teraz użyj pliku:  {ONNX_FILE.name}")