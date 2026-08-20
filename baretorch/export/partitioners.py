# baretorch/export/partitioners.py
from typing import Optional, List, Any


def get_backend_partitioner(backend_name: str) -> Optional[List[Any]]:
    """
    Resolves target backend partitioner delegates for ExecuTorch EXIR graph lowering:
      - 'xnnpack' : Universal ARM / x86 CPU delegate (KleidiAI)
      - 'coreml'  : Apple Neural Engine (ANE) and Metal GPU
      - 'qnn'     : Qualcomm Snapdragon Hexagon NPU & Adreno GPU
      - 'vulkan'  : Mobile GPU Compute Shaders (Adreno / Mali)
      - 'none'    : Native EXIR Edge Dialect (no partitioning)
    """
    backend = backend_name.lower().strip()

    if backend in ["none", "eager"]:
        print("  🎯 Backend: Native ExecuTorch EXIR (Eager / Unpartitioned)")
        return None

    elif backend == "xnnpack":
        print("  🎯 Backend Delegate: XNNPACK (ARM CPU / KleidiAI Vector Engine)")
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            return [XnnpackPartitioner()]
        except ImportError:
            print("  ⚠️ XnnpackPartitioner module not found. Falling back to native EXIR IR.")
            return None

    elif backend in ["coreml", "metal", "ane"]:
        print("  🎯 Backend Delegate: CoreML / Metal (Apple Neural Engine & Apple GPU)")
        
        # 1. Primary ExecuTorch CoreML Partitioner import path
        try:
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
            return [CoreMLPartitioner()]
        except ImportError:
            # 2. Alternative module subpath fallback
            try:
                from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
                return [CoreMLPartitioner()]
            except ImportError:
                try:
                    from executorch.backends.apple.coreml.compiler import CoreMLPartitioner
                    return [CoreMLPartitioner()]
                except ImportError:
                    print("  ⚠️ CoreML Partitioner not available. Install coremltools and ExecuTorch Apple bindings.")
                    return None

    elif backend in ["qnn", "qualcomm", "hexagon"]:
        print("  🎯 Backend Delegate: Qualcomm QNN (Hexagon NPU & Adreno GPU)")
        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
            return [QnnPartitioner()]
        except ImportError:
            print("  ⚠️ QnnPartitioner not available. Ensure Qualcomm QNN SDK is installed.")
            return None

    elif backend == "vulkan":
        print("  🎯 Backend Delegate: Vulkan (Mobile GPU Compute Shaders)")
        try:
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
            return [VulkanPartitioner()]
        except ImportError:
            print("  ⚠️ VulkanPartitioner not available. Falling back to native EXIR IR.")
            return None

    else:
        print(f"  ⚠️ Unknown backend '{backend_name}'. Proceeding without partitioner...")
        return None