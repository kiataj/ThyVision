from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HippoCanFly/ct-thyroid-classifier",
    repo_type="model",
    local_dir="/app/hf_model",
    token="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"
)
