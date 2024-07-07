""" CuPy Conway's Game of Life

USAGE

    modal run ./modal-cupy-conway.py

NOTES

    The first attempt was to use `debian_slim` and install `cupy-cuda12x`:

      image = modal.Image.debian_slim().pip_install("cupy-cuda12x")

    However, that resulted in the following error:

      CuPy failed to load libnvrtc.so.12: OSError: libnvrtc.so.12: cannot open shared object file: No such file or directory        │

    Then, the offical cuda Docker image was used as outlined in the Modal guide:

        https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image

"""

import modal

app = modal.App()

cuda_version = "12.4.0"
flavor = "devel"
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch",
    )
    .run_commands(  # add flash-attn
        "pip install cupy-cuda12x --no-build-isolation"
    )
)


ITERATIONS = 1_000
UNIVERSE_DIMENSION_M = 10_000
UNIVERSE_DIMENSION_N = 10_000


@app.function(gpu="any", image=image)
def run_cupy_conways_game_of_life():
    import cupy as cp
    import cupyx.scipy.ndimage as nd

    kernel = cp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=cp.uint8)

    universe = cp.random.randint(2, size=(UNIVERSE_DIMENSION_M, UNIVERSE_DIMENSION_N))

    for _ in range(ITERATIONS):
        neighbor_counts = nd.convolve(universe, kernel, mode="constant")

        universe = (
            universe & cp.isin(neighbor_counts, cp.array([2, 3])) # cell survived
        ) | (
            ~universe & (neighbor_counts == 3)                    # cell was birthed
        )

    print(universe)
