from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def hf_download(repo: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    hf_repo = "FunAudioLLM/CosyVoice2-0.5B" if repo == "iic/CosyVoice2-0.5B" else repo
    allow = None
    ignore = ["*.h5", "*.ot", "*.msgpack", "*.onnx"]
    if repo == "iic/CosyVoice2-0.5B":
        allow = ["cosyvoice2.yaml", "campplus.onnx", "speech_tokenizer_v2.onnx", "spk2info.pt", "llm.pt", "flow.pt", "hift.pt", "README.md", "configuration.json", "config.json"]
        ignore = None
    snapshot_download(
        repo_id=hf_repo,
        local_dir=target,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )


def ms_download(repo: str, target: Path) -> None:
    from modelscope import snapshot_download

    allow = None
    ignore = ["*.h5", "*.ot", "*.msgpack", "*.onnx"]
    if repo == "iic/CosyVoice2-0.5B":
        allow = ["cosyvoice2.yaml", "campplus.onnx", "speech_tokenizer_v2.onnx", "spk2info.pt", "llm.pt", "flow.pt", "hift.pt", "README.md", "configuration.json", "config.json"]
        ignore = None
    snapshot_download(
        repo,
        local_dir=str(target),
        allow_patterns=allow,
        ignore_patterns=ignore,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 AutoVidDub 推荐模型")
    parser.add_argument("--source", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    download = ms_download if args.source == "modelscope" else hf_download

    jobs = [("Helsinki-NLP/opus-mt-zh-en", MODELS / "opus-mt-zh-en")]
    if not args.skip_asr:
        jobs.extend(
            [
                ("Systran/faster-whisper-small", MODELS / "faster-whisper-small"),
                ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", MODELS / "faster-whisper-large-v3-turbo"),
            ]
        )
    if not args.skip_tts:
        jobs.append(("iic/CosyVoice2-0.5B", MODELS / "CosyVoice2-0.5B"))

    for repo, target in jobs:
        required = [target / "config.json"]
        if repo == "iic/CosyVoice2-0.5B":
            required = [target / name for name in ("cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v2.onnx")]
        if all(path.exists() and path.stat().st_size > 0 for path in required):
            print(f"[跳过] {repo} 已存在")
            continue
        print(f"[下载] {repo} -> {target}")
        download(repo, target)
    print("模型下载完成。")


if __name__ == "__main__":
    main()
