import os
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.adversarial_ai.attacker_model import AttackerModel
from models.regime.lstm_classifier import LSTMRegimeClassifier

SEQ_LEN = 60
N_FEATS = 57
DEVICE = "cpu"

def verify_onnx_model(onnx_path, dummy_input, pytorch_output):
    """Verify that ONNX model predictions match PyTorch outputs."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        # Format dummy input to numpy
        if isinstance(dummy_input, tuple):
            ort_inputs = {session.get_inputs()[i].name: dummy_input[i].numpy() for i in range(len(dummy_input))}
        else:
            ort_inputs = {input_name: dummy_input.numpy()}
            
        ort_outs = session.run(None, ort_inputs)
        onnx_output = ort_outs[0]
        
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        if max_diff < 1e-4:
            print(f"  ✓ ONNX Verification PASS (max absolute difference: {max_diff:.2e})")
        else:
            print(f"  ⚠ ONNX Verification WARNING (max absolute difference: {max_diff:.2e})")
    except ImportError:
        print("  ONNX Runtime not installed, skipping verification step.")
    except Exception as e:
        print(f"  ✗ ONNX Verification FAILED: {e}")

def main():
    print("================================================================================")
    print("                NEURAL ENGINE ONNX EXPORT & COMPILATION SYSTEM                  ")
    print("================================================================================")

    # 1. Export TemporalFusionModel
    print("\n[1/4] Exporting TemporalFusionModel...")
    pt_path = "saved_models/temporal_model.pt"
    onnx_path = "saved_models/temporal_model.onnx"
    if os.path.exists(pt_path):
        try:
            model_wrapper = TemporalFusionModel(name="temporal_fusion")
            model_wrapper.load(pt_path)
            model = model_wrapper.model.to(DEVICE)
            model.eval()

            dummy_x = torch.randn(1, SEQ_LEN, N_FEATS, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                py_out = model(dummy_x).cpu().numpy()

            torch.onnx.export(
                model,
                dummy_x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"  ✓ Successfully exported to: {onnx_path}")
            verify_onnx_model(onnx_path, dummy_x, py_out)
        except Exception as e:
            print(f"  ✗ Failed to export TemporalFusionModel: {e}")
    else:
        print(f"  ✗ Source checkpoint {pt_path} not found.")

    # 2. Export MAMLModel
    print("\n[2/4] Exporting MAMLModel...")
    pt_path = "saved_models/maml_model.pt"
    onnx_path = "saved_models/maml_model.onnx"
    if os.path.exists(pt_path):
        try:
            model_wrapper = MAMLModel(name="maml")
            model_wrapper.load(pt_path)
            model = model_wrapper.model.to(DEVICE)
            model.eval()

            # MAMLNetwork forward expects: shape [batch, seq_len, d_feat] or [batch, d_feat]
            dummy_x = torch.randn(1, SEQ_LEN, N_FEATS, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                py_out = model(dummy_x).cpu().numpy()

            torch.onnx.export(
                model,
                dummy_x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"  ✓ Successfully exported to: {onnx_path}")
            verify_onnx_model(onnx_path, dummy_x, py_out)
        except Exception as e:
            print(f"  ✗ Failed to export MAMLModel: {e}")
    else:
        print(f"  ✗ Source checkpoint {pt_path} not found.")

    # 3. Export AttackerModel
    print("\n[3/4] Exporting AttackerModel...")
    pt_path = "saved_models/adversarial_attacker.pt"
    onnx_path = "saved_models/adversarial_attacker.onnx"
    if os.path.exists(pt_path):
        try:
            model_wrapper = AttackerModel(name="adversarial_attacker")
            model_wrapper.load("saved_models/adversarial_attacker")
            model = model_wrapper.model.to(DEVICE)
            model.eval()

            dummy_x = torch.randn(1, N_FEATS, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                py_out = model(dummy_x).cpu().numpy()

            torch.onnx.export(
                model,
                dummy_x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"  ✓ Successfully exported to: {onnx_path}")
            verify_onnx_model(onnx_path, dummy_x, py_out)
        except Exception as e:
            print(f"  ✗ Failed to export AttackerModel: {e}")
    else:
        print(f"  ✗ Source checkpoint {pt_path} not found.")

    # 4. Export LSTMRegimeClassifier
    print("\n[4/4] Exporting LSTMRegimeClassifier...")
    pt_path = "saved_models/regime_ensemble.pkl.lstm"
    onnx_path = "saved_models/regime_ensemble.lstm.onnx"
    if os.path.exists(pt_path):
        try:
            model_wrapper = LSTMRegimeClassifier(name="lstm_regime")
            model_wrapper.load(pt_path)
            model = model_wrapper.model.to(DEVICE)
            model.eval()

            # Regime model is trained on 4 HMM features
            dummy_x = torch.randn(1, SEQ_LEN, 4, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                py_out = model(dummy_x).cpu().numpy()

            torch.onnx.export(
                model,
                dummy_x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=14,
                do_constant_folding=True
            )
            print(f"  ✓ Successfully exported to: {onnx_path}")
            verify_onnx_model(onnx_path, dummy_x, py_out)
        except Exception as e:
            print(f"  ✗ Failed to export LSTMRegimeClassifier: {e}")
    else:
        print(f"  ✗ Source checkpoint {pt_path} not found.")

if __name__ == "__main__":
    main()
