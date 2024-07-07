"""
USAGE

    modal run modal-cuda-hello-world.py

"""

import modal

app = modal.App()


@app.function(gpu="any")
def check_nvidia_smi():
    import subprocess

    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version: 550.54.15" in output
    assert "CUDA Version: 12.4" in output
    return output


@app.local_entrypoint()
def main():
    print(check_nvidia_smi.remote())
