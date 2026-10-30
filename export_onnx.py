"""
ONNX Export and Benchmarking Tool for RetiSpec
Exports ConvNeXtDR from PyTorch (.pth) to ONNX (.onnx) format.
Validates outputs and benchmarks CPU inference performance.
"""
import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort

import config as cfg
from model import ConvNeXtDR


def export_to_onnx(
    checkpoint_path: str = cfg.BEST_MODEL_PATH,
    output_onnx_path: str = cfg.ONNX_MODEL_PATH,
    opset_version: int = 14,
    benchmark_runs: int = 50
):
    print("=" * 70)
    print("RetiSpec: PyTorch (.pth) -> ONNX (.onnx) Exporter")
    print("=" * 70)
    
    # 1. Load PyTorch model on CPU
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = ConvNeXtDR(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Load EMA weights if available, else standard model_state
    state_dict = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    print("PyTorch model loaded successfully on CPU.")

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Saved ONNX model to: {output_onnx_path}")
    onnx_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    pth_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Model File Size: PyTorch .pth = {pth_size_mb:.1f} MB -> ONNX = {onnx_size_mb:.1f} MB")

    # 3. Check ONNX Model Integrity
    print("\nVerifying ONNX model integrity...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model passed structural verification!")

    # 4. Compare PyTorch vs ONNX Runtime Numerical Precision
    print("\nValidating numerical parity between PyTorch & ONNX Runtime...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(output_onnx_path, session_options, providers=["CPUExecutionProvider"])

    test_input = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    with torch.no_grad():
        pt_logits = model(test_input).numpy()

    ort_inputs = {"input": test_input.numpy()}
    ort_logits = session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"Maximum absolute output difference: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Parity mismatch: max difference {max_diff} exceeds tolerance!"
    print("Parity confirmed: PyTorch and ONNX Runtime predictions match perfectly!")

    # 5. Benchmark Latency (PyTorch CPU vs ONNX Runtime CPU)
    print(f"\nBenchmarking inference speed over {benchmark_runs} runs on CPU...")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_input)
        _ = session.run(None, ort_inputs)

    # PyTorch CPU timing
    t0 = time.perf_counter()
    for _ in range(benchmark_runs):
        with torch.no_grad():
            _ = model(test_input)
    pt_avg_ms = ((time.perf_counter() - t0) / benchmark_runs) * 1000

    # ONNX Runtime CPU timing
    t0 = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = session.run(None, ort_inputs)
    ort_avg_ms = ((time.perf_counter() - t0) / benchmark_runs) * 1000

    speedup = pt_avg_ms / ort_avg_ms
    print("-" * 50)
    print(f"PyTorch CPU Latency:      {pt_avg_ms:.2f} ms / image")
    print(f"ONNX Runtime CPU Latency: {ort_avg_ms:.2f} ms / image")
    print(f"Speedup via ONNX Runtime: {speedup:.2f}x faster on CPU!")
    print("-" * 50)

    return output_onnx_path


if __name__ == "__main__":
    export_to_onnx()

