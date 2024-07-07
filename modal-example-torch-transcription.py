"""
USAGE

    modal run ./modal-example-torch-transcription.py

EXAMPLE OUTPUT

    ✓ Created objects.
    ├── 🔨 Created mount /Users/colton/src/modal/modal-example-torch-transcription.py
    └── 🔨 Created function run_transformers.
    /usr/local/lib/python3.12/site-packages/transformers/models/whisper/generation_whisper.py:480: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
      warnings.warn(
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Stopping app - local entrypoint completed.
     I have a dream that one day this nation will rise up live out the true meaning of its creed
    Runner terminated.
    ✓ App completed. View run at https://modal.com/cmpadden/main/apps/ap-4sSXSXNs8SLwpagMKNByZj

"""

import modal

app = modal.App()

image = modal.Image.debian_slim().pip_install("transformers[torch]")
image = image.apt_install("ffmpeg")  # for audio processing


@app.function(gpu="any", image=image)
def run_transformers():
    from transformers import pipeline

    transcriber = pipeline(model="openai/whisper-tiny.en", device="cuda")
    result = transcriber("https://modal-public-assets.s3.amazonaws.com/mlk.flac")
    print(
        result["text"]
    )  # I have a dream that one day this nation will rise up live out the true meaning of its creed
