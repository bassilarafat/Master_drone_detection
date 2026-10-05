"""Report the local Python, computer-vision and CUDA environment."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    """Print dependency versions and the available CUDA device."""
    import cv2
    import numpy as np
    import skimage
    import torch

    print("=" * 50)
    print("MASTER DRONE DETECTION ENVIRONMENT")
    print("=" * 50)

    print("Python:", sys.version)
    print("Platform:", platform.platform())

    print("\nLibraries:")
    print("PyTorch:", torch.__version__)
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("scikit-image:", skimage.__version__)

    print("\nCUDA:")
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM GB:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        )

    print("=" * 50)


if __name__ == "__main__":
    main()
