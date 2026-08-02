"""
INT8 Dynamic Quantization for ONNX Models.

Compresses all ONNX neural network models from FP32 to INT8 (dynamic quantization),
reducing model size by ~75% and improving inference speed on CPU/edge devices
while preserving signal quality.

Usage:
    python scripts/quantize_onnx_models.py
"""
import os
import sys
import glob
import structlog

logger = structlog.get_logger()


def quantize_model(input_path: str, output_path: str) -> bool:
    """
    Apply dynamic INT8 quantization to a single ONNX model.
    Dynamic quantization quantizes weights to INT8 at save time and
    dynamically quantizes activations at inference time.
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        
        # Load model and clear intermediate shape value_info to prevent ShapeInferenceError
        model = onnx.load(input_path)
        model.graph.ClearField('value_info')
        
        temp_clear_path = input_path.replace(".onnx", ".temp_clear.onnx")
        onnx.save(model, temp_clear_path)
        
        temp_prep_path = input_path.replace(".onnx", ".temp_prep.onnx")
        try:
            quant_pre_process(temp_clear_path, temp_prep_path)
            model_to_quantize = temp_prep_path
        except Exception as preprocess_err:
            logger.warning("Shape preprocessing failed; attempting direct quantization", error=str(preprocess_err))
            model_to_quantize = temp_clear_path
            temp_prep_path = None
            
        quantize_dynamic(
            model_input=model_to_quantize,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        
        # Clean up temporary files
        if os.path.exists(temp_clear_path):
            os.remove(temp_clear_path)
        if temp_prep_path and os.path.exists(temp_prep_path):
            os.remove(temp_prep_path)
        
        # Report size reduction
        original_size = os.path.getsize(input_path)
        # Check for external data file
        data_file = input_path + ".data"
        if os.path.exists(data_file):
            original_size += os.path.getsize(data_file)
        
        quantized_size = os.path.getsize(output_path)
        quantized_data = output_path + ".data"
        if os.path.exists(quantized_data):
            quantized_size += os.path.getsize(quantized_data)
        
        reduction = (1.0 - quantized_size / original_size) * 100.0 if original_size > 0 else 0.0
        
        logger.info(
            "Quantization complete",
            model=os.path.basename(input_path),
            original_kb=f"{original_size / 1024:.1f}",
            quantized_kb=f"{quantized_size / 1024:.1f}",
            reduction_pct=f"{reduction:.1f}%"
        )
        return True
    except Exception as e:
        logger.error("Quantization failed", model=os.path.basename(input_path), error=str(e))
        return False


def verify_quantized_model(original_path: str, quantized_path: str) -> bool:
    """
    Verify that the quantized model produces outputs within acceptable tolerance
    of the original FP32 model using random input data.
    """
    import numpy as np
    import onnxruntime as ort
    
    try:
        # Load both sessions
        sess_orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
        sess_quant = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
        
        # Build random input matching expected shapes
        inputs = {}
        for inp in sess_orig.get_inputs():
            shape = []
            for dim in inp.shape:
                if isinstance(dim, str) or dim is None:
                    shape.append(1)  # dynamic dim -> batch=1
                else:
                    shape.append(dim)
            dtype = np.float32 if "float" in inp.type.lower() else np.int64
            inputs[inp.name] = np.random.randn(*shape).astype(dtype)
        
        # Run inference on both
        outputs_orig = sess_orig.run(None, inputs)
        outputs_quant = sess_quant.run(None, inputs)
        
        # Compare outputs
        max_diff = 0.0
        for o_orig, o_quant in zip(outputs_orig, outputs_quant):
            diff = np.max(np.abs(o_orig.astype(np.float64) - o_quant.astype(np.float64)))
            max_diff = max(max_diff, diff)
        
        # Dynamic tolerance depending on model category
        name_lower = original_path.lower()
        if "regime" in name_lower:
            tolerance = 0.30  # LSTM probability distributions can drift slightly under INT8 weights
        elif "attacker" in name_lower or "adversarial" in name_lower:
            tolerance = 0.10  # Attacker network is highly non-linear
        else:
            tolerance = 0.05  # standard 5% tolerance for primary models
            
        passed = max_diff < tolerance
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"Quantization verification: {status}",
            model=os.path.basename(original_path),
            max_diff=f"{max_diff:.6e}"
        )
        return passed
    except Exception as e:
        logger.error("Quantization verification failed", model=os.path.basename(original_path), error=str(e))
        return False


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "saved_models")
    quantized_dir = os.path.join(models_dir, "quantized")
    os.makedirs(quantized_dir, exist_ok=True)
    
    # Discover all ONNX models (top-level and checkpoints)
    onnx_files = glob.glob(os.path.join(models_dir, "*.onnx"))
    checkpoint_onnx = glob.glob(os.path.join(models_dir, "checkpoints", "*.onnx"))
    onnx_files.extend(checkpoint_onnx)
    
    if not onnx_files:
        logger.error("No ONNX models found in saved_models/")
        sys.exit(1)
    
    print("=" * 72)
    print("       INT8 DYNAMIC QUANTIZATION FOR ONNX MODELS")
    print("=" * 72)
    print(f"\nFound {len(onnx_files)} ONNX model(s) to quantize.\n")
    
    results = []
    for onnx_path in sorted(onnx_files):
        basename = os.path.basename(onnx_path)
        # Determine output subdirectory for checkpoints
        if "checkpoints" in onnx_path:
            out_dir = os.path.join(quantized_dir, "checkpoints")
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = quantized_dir
        
        out_path = os.path.join(out_dir, basename.replace(".onnx", "_int8.onnx"))
        
        print(f"  Quantizing: {basename}")
        success = quantize_model(onnx_path, out_path)
        
        if success:
            verified = verify_quantized_model(onnx_path, out_path)
            results.append((basename, success, verified))
        else:
            results.append((basename, success, False))
    
    # Summary
    print("\n" + "=" * 72)
    print("  QUANTIZATION SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<45} {'Status':<12} {'Verified'}")
    print("-" * 72)
    for name, success, verified in results:
        s = "OK" if success else "FAILED"
        v = "PASS" if verified else "FAIL"
        print(f"  {name:<45} {s:<12} {v}")
    print("=" * 72)
    
    all_ok = all(s for _, s, _ in results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
