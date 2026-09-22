from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gradio as gr
import imageio_ffmpeg
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
HISTORY_FILE = OUTPUTS / "task_history.json"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_CANCEL_EVENT = threading.Event()
_ASR_CACHE: dict[str, object] = {}
_TRANSLATOR_CACHE: dict[str, tuple[object, object]] = {}
_TTS_CACHE: dict[str, object] = {}
_CURRENT_STATUS = "等待任务"
_TASK_PROGRESS = threading.local()
_LAST_MEDIA: str | None = None

LANGUAGES = {
    "自动检测": ("自动", "auto", "自动"),
    "中文": ("zh", "zho_Hans", "Chinese"),
    "英语": ("en", "eng_Latn", "English"),
    "日语": ("ja", "jpn_Jpan", "Japanese"),
    "韩语": ("ko", "kor_Hang", "Korean"),
}


def console_log(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_message = str(message).encode(encoding, errors="replace").decode(encoding)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {safe_message}", flush=True)


def set_status(message: str) -> None:
    global _CURRENT_STATUS
    with _STATE_LOCK:
        _CURRENT_STATUS = message
    reporter = getattr(_TASK_PROGRESS, "reporter", None)
    if reporter:
        reporter(stage=message.splitlines()[-1])


def report_task_progress(stage: str, completed: int | None = None, total: int | None = None) -> None:
    reporter = getattr(_TASK_PROGRESS, "reporter", None)
    if reporter:
        reporter(stage=stage, completed=completed, total=total)


def get_status() -> str:
    with _STATE_LOCK:
        return _CURRENT_STATUS


def request_cancel() -> str:
    _CANCEL_EVENT.set()
    message = "已请求停止，将在当前步骤结束后停止。"
    set_status(message)
    console_log("Cancel requested from Web UI")
    return message


def check_cancelled() -> None:
    if _CANCEL_EVENT.is_set():
        raise RuntimeError("任务已由用户停止。")


def restore_page_state() -> tuple[str | None, str, list[list[str]]]:
    media = _LAST_MEDIA if _LAST_MEDIA and Path(_LAST_MEDIA).exists() else None
    return media, get_status(), load_history()


def poll_dashboard() -> tuple[str, list[list[str]]]:
    return get_status(), load_history()


def load_history() -> list[list[str]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        items = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return [[x.get(k, "") for k in ("time", "file", "languages", "status", "output")] for x in items[:30]]
    except Exception:
        return []


def add_history(file_name: str, languages: str, status: str, output: str = "") -> None:
    rows = []
    if HISTORY_FILE.exists():
        try:
            rows = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    rows.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "file": file_name,
        "languages": languages,
        "status": status,
        "output": output,
    })
    HISTORY_FILE.write_text(json.dumps(rows[:100], ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Segment:
    start: float
    end: float
    source: str
    target: str = ""


def run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "FFmpeg 执行失败")


def extract_audio(media: str, output: Path) -> None:
    run_ffmpeg(["-i", media, "-vn", "-ac", "2", "-ar", "44100", str(output)])


def separate_audio(source: Path, job: Path) -> tuple[Path, Path]:
    stems_root = job / "stems"
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-d",
        "cuda" if _cuda_available() else "cpu",
        "-o",
        str(stems_root),
        str(source),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.stdout.strip():
        console_log(proc.stdout.strip())
    if proc.returncode:
        raise RuntimeError(f"人声分离失败：{proc.stderr.strip()}")
    result_dir = stems_root / "htdemucs" / source.stem
    vocals = result_dir / "vocals.wav"
    background = result_dir / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        raise RuntimeError("Demucs 没有生成预期的人声和背景音轨。")
    return vocals, background


def prepare_voice_reference(
    source: Path,
    output: Path,
    window_seconds: float = 6.0,
    preferred_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Select a speech-dense prompt and normalize it for voice cloning."""
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(str(source), sr=24000, mono=True)
    if audio.size < sample_rate * 2:
        raise RuntimeError("声音参考太短，请提供至少 3 秒清晰、无背景音乐的人声。")

    hop_length = 240
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length)[0]
    window_frames = min(len(rms), max(1, round(window_seconds * sample_rate / hop_length)))
    noise_floor = float(np.percentile(rms, 30))
    speech_level = float(np.percentile(rms, 85))
    threshold = max(0.002, noise_floor * 2.5, speech_level * 0.22)
    active = (rms >= threshold).astype(np.float32)
    energy = np.log1p(rms / max(threshold, 1e-6))
    if preferred_range:
        start, requested_duration = preferred_range
        duration = min(window_seconds, requested_duration, len(audio) / sample_rate - start)
        best_frame = max(0, round(start * sample_rate / hop_length))
        window_frames = max(1, round(duration * sample_rate / hop_length))
    else:
        score = np.convolve(active + 0.20 * energy, np.ones(window_frames), mode="valid")
        best_frame = int(np.argmax(score)) if score.size else 0
        start = best_frame * hop_length / sample_rate
        duration = min(window_seconds, len(audio) / sample_rate - start)
    active_ratio = float(active[best_frame : best_frame + window_frames].mean())
    if active_ratio < 0.35:
        raise RuntimeError("没有找到足够清晰的连续人声，请上传 4-8 秒无音乐、无混响的声音参考。")

    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-af", "highpass=f=70,loudnorm=I=-20:TP=-2:LRA=7",
        "-ac", "1", "-ar", "24000", str(output),
    ])
    return start, active_ratio


def load_asr(model_name: str):
    from faster_whisper import WhisperModel

    local = MODELS / f"faster-whisper-{model_name}"
    if not (local / "model.bin").exists():
        raise FileNotFoundError(
            f"缺少 ASR 模型 faster-whisper-{model_name}。请运行 download_models.py --source modelscope。"
        )
    key = str(local)
    if key not in _ASR_CACHE:
        console_log(f"Loading ASR model: {model_name}")
        _ASR_CACHE.clear()
        _ASR_CACHE[key] = WhisperModel(
            key,
            device="cuda" if _cuda_available() else "cpu",
            compute_type="float16" if _cuda_available() else "int8",
        )
        console_log("ASR model loaded")
    return _ASR_CACHE[key]


def transcribe(audio: Path, model_name: str, language: str) -> list[Segment]:
    model = load_asr(model_name)
    segments, _ = model.transcribe(
        str(audio), language=None if language == "自动" else language, vad_filter=True, beam_size=5
    )
    result = [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments if s.text.strip()]
    if not result:
        raise RuntimeError("没有识别到语音，请检查输入文件。")
    return result


def merge_short_segments(segments: list[Segment], max_duration: float = 10.0, max_gap: float = 0.7) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            combined_duration = segment.end - previous.start
            gap = segment.start - previous.end
            if gap <= max_gap and combined_duration <= max_duration and (previous.end - previous.start < 3.0):
                previous.end = segment.end
                previous.source = f"{previous.source} {segment.source}".strip()
                continue
        merged.append(Segment(segment.start, segment.end, segment.source))
    return merged


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def translate_local(texts: list[str], source_language: str, target_language: str) -> list[str]:
    if source_language in {"自动", "zh"} and target_language == "英语":
        return translate_marian_zh_en(texts)
    return translate_nllb(texts, source_language, target_language)


def translate_marian_zh_en(texts: list[str]) -> list[str]:
    from transformers import MarianMTModel, MarianTokenizer

    local = MODELS / "opus-mt-zh-en"
    # Some Windows tokenizer loaders mishandle absolute paths containing CJK characters.
    model_id = local.relative_to(ROOT).as_posix() if local.exists() else "Helsinki-NLP/opus-mt-zh-en"
    if model_id not in _TRANSLATOR_CACHE:
        console_log("Loading local translation model: opus-mt-zh-en")
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
        if _cuda_available():
            model = model.cuda()
        _TRANSLATOR_CACHE[model_id] = (tokenizer, model)
        console_log("Translation model loaded")
    tokenizer, model = _TRANSLATOR_CACHE[model_id]
    output: list[str] = []
    for pos in range(0, len(texts), 16):
        batch = tokenizer(texts[pos : pos + 16], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        generated = model.generate(**batch, max_new_tokens=256, num_beams=4)
        output.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [clean_translation(text, source) for text, source in zip(output, texts)]


def translate_nllb(texts: list[str], source_language: str, target_language: str) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    local = MODELS / "nllb-200-distilled-600M"
    if not (local / "config.json").exists():
        raise FileNotFoundError("尚未安装 NLLB-200 多语言模型，请在“模型管理”中下载。")
    source_key = "中文" if source_language in {"自动", "zh"} else source_language
    if source_key not in LANGUAGES or target_language not in LANGUAGES:
        raise ValueError("不支持所选语言方向。")
    source_code = LANGUAGES[source_key][1]
    target_code = LANGUAGES[target_language][1]
    key = f"nllb:{local.resolve()}"
    if key not in _TRANSLATOR_CACHE:
        console_log("Loading local translation model: NLLB-200 distilled 600M")
        tokenizer = AutoTokenizer.from_pretrained(str(local), src_lang=source_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(local))
        if _cuda_available():
            model = model.cuda()
        _TRANSLATOR_CACHE.clear()
        _TRANSLATOR_CACHE[key] = (tokenizer, model)
    tokenizer, model = _TRANSLATOR_CACHE[key]
    tokenizer.src_lang = source_code
    output: list[str] = []
    for pos in range(0, len(texts), 8):
        batch = tokenizer(texts[pos : pos + 8], return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        generated = model.generate(
            **batch,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_new_tokens=256,
            num_beams=4,
        )
        output.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [clean_translation(text, source) for text, source in zip(output, texts)]


def download_nllb() -> str:
    from huggingface_hub import snapshot_download

    target = MODELS / "nllb-200-distilled-600M"
    set_status("正在下载 NLLB-200 多语言模型，请保持终端开启。")
    console_log("Downloading NLLB-200 distilled 600M")
    snapshot_download(repo_id="facebook/nllb-200-distilled-600M", local_dir=target)
    message = "NLLB-200 多语言模型安装完成。"
    set_status(message)
    console_log(message)
    return message


def model_status() -> list[list[str]]:
    checks = [
        ("Whisper large-v3-turbo", MODELS / "faster-whisper-large-v3-turbo"),
        ("OPUS-MT 中译英", MODELS / "opus-mt-zh-en"),
        ("NLLB-200 多语言", MODELS / "nllb-200-distilled-600M"),
        ("CosyVoice2 0.5B", MODELS / "CosyVoice2-0.5B"),
    ]
    return [[name, "已安装" if path.exists() else "未安装", str(path)] for name, path in checks]


def clean_translation(text: str, source: str) -> str:
    """Remove model loops before they reach TTS."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    repeated_word = re.compile(r"\b([A-Za-z]+(?:'[A-Za-z]+)?)(?:\s*[,;:]?\s+\1\b){2,}", re.IGNORECASE)
    while True:
        collapsed = repeated_word.sub(r"\1", cleaned)
        if collapsed == cleaned:
            break
        cleaned = collapsed
    # Marian occasionally loops until max tokens. Keep a concise sentence instead
    # of sending hundreds of repeated words to the speech model.
    words = cleaned.split()
    source_chars = len(re.sub(r"[\s\W]+", "", source, flags=re.UNICODE))
    limit = max(12, source_chars * 2 + 4)
    if len(words) > limit:
        cleaned = " ".join(words[:limit]).rstrip(",;:") + "."
        console_log(f"Trimmed abnormal translation: {len(words)} words -> {limit} words")
    return cleaned


def translate_api(
    texts: list[str], base_url: str, api_key: str, model: str, target_language: str, style: str
) -> list[str]:
    if not base_url or not api_key or not model:
        raise ValueError("第三方中转平台需要完整填写 Base URL、API Key 和模型名称。")
    url = base_url.rstrip("/")
    if not url.endswith("chat/completions"):
        url += "/chat/completions"

    def request_batch(batch: list[str]) -> list[str]:
        payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Translate every subtitle into {LANGUAGES[target_language][2]}. "
                        f"Style: {style}. Keep each line concise for dubbing. "
                        f"Return only one valid JSON array containing exactly {len(batch)} strings. "
                        "Do not merge, omit, number, or explain any item."
                    ),
                },
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
        }
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:500].strip()
            raise RuntimeError(f"第三方翻译请求失败（HTTP {response.status_code}）：{detail}") from exc
        content = response.json()["choices"][0]["message"]["content"].strip()
        translated = None
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char != "[":
                continue
            try:
                value, _ = decoder.raw_decode(content[index:])
                if isinstance(value, list):
                    translated = value
                    break
            except json.JSONDecodeError:
                continue
        if translated is None:
            raise RuntimeError("第三方翻译接口未返回有效的 JSON 字幕数组。")
        return [str(item) for item in translated]

    results: list[str] = []
    batch_size = 10
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        batch_number = offset // batch_size + 1
        console_log(f"Translating API batch {batch_number}/{total_batches} ({len(batch)} subtitles)")
        translated = request_batch(batch)
        if len(translated) != len(batch):
            console_log(
                f"API batch {batch_number} returned {len(translated)}/{len(batch)} subtitles; retrying individually"
            )
            translated = []
            for source in batch:
                single = request_batch([source])
                if len(single) != 1:
                    raise RuntimeError("第三方翻译接口单句补翻仍未返回一条字幕。")
                translated.append(single[0])
        results.extend(clean_translation(item, source) for item, source in zip(translated, batch))
    if len(results) != len(texts):
        raise RuntimeError(f"翻译结果校验失败：输入 {len(texts)} 句，输出 {len(results)} 句。")
    return results


def load_cosyvoice(model_path: str):
    cosy_root = ROOT / "vendor" / "CosyVoice"
    matcha_root = cosy_root / "third_party" / "Matcha-TTS"
    for item in (cosy_root, matcha_root):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    import librosa
    import torch
    import cosyvoice.cli.frontend as cosy_frontend
    from cosyvoice.cli.cosyvoice import CosyVoice2

    def load_audio_compat(wav, target_sr, min_sr=16000):
        audio, source_sr = librosa.load(str(wav), sr=None, mono=True)
        if source_sr < min_sr:
            raise ValueError(f"参考音频采样率 {source_sr} 低于 {min_sr}")
        if source_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
        return torch.from_numpy(audio).unsqueeze(0)

    # Torchaudio 2.11 requires a system FFmpeg DLL build on Windows. The
    # application already ships a decoder, so use librosa for prompt loading.
    cosy_frontend.load_wav = load_audio_compat

    local = Path(model_path.strip()) if model_path.strip() else MODELS / "CosyVoice2-0.5B"
    if not local.exists():
        raise FileNotFoundError("未找到 CosyVoice2-0.5B，请运行 download_models.py 下载模型。")
    key = str(local.resolve())
    if key not in _TTS_CACHE:
        console_log(f"Loading CosyVoice2 model: {local}")
        _TTS_CACHE.clear()
        # FP32 is slower but avoids unstable flow-model output on newer Windows CUDA builds.
        _TTS_CACHE[key] = CosyVoice2(key, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        console_log("CosyVoice2 model loaded")
    return _TTS_CACHE[key]


def synth_cosyvoice(
    text: str,
    output: Path,
    ref_audio: Path,
    model_path: str,
    target_seconds: float,
    emotion: str = "自然",
) -> None:
    import torch

    model = load_cosyvoice(model_path)
    word_count = max(1, len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)))
    max_reasonable = max(3.0, target_seconds * 2.0, word_count / 1.7 + 1.8)
    candidates: list[tuple[float, object]] = []
    for attempt in range(3):
        torch.manual_seed(20260920 + attempt)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260920 + attempt)
        if emotion and emotion != "自然":
            instruction = f"用{emotion}的语气说这句话<|endofprompt|>"
            generator = model.inference_instruct2(text.strip(), instruction, str(ref_audio), stream=False)
        else:
            generator = model.inference_cross_lingual(text.strip(), str(ref_audio), stream=False)
        chunks = [item["tts_speech"].cpu() for item in generator]
        if not chunks:
            continue
        wav = torch.cat(chunks, dim=1).squeeze(0).numpy()
        duration = len(wav) / model.sample_rate
        candidates.append((duration, wav))
        if duration <= max_reasonable:
            break
        console_log(
            f"CosyVoice abnormal duration {duration:.2f}s (limit {max_reasonable:.2f}s); retry {attempt + 1}/3"
        )
    if not candidates:
        raise RuntimeError("CosyVoice2 没有生成音频。")
    duration, wav = min(candidates, key=lambda item: item[0])
    if duration > max_reasonable:
        console_log(f"Using shortest CosyVoice result after retries: {duration:.2f}s")
    sf.write(output, wav, model.sample_rate)


def synth_http(text: str, output: Path, base_url: str, api_key: str, model: str, voice: str) -> None:
    if not base_url or not model:
        raise ValueError("第三方 TTS 需要 Base URL 和模型名称。")
    url = base_url.rstrip("/")
    if not url.endswith("audio/speech"):
        url += "/audio/speech"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "voice": voice or "alloy", "input": text, "response_format": "wav"},
        timeout=300,
    )
    response.raise_for_status()
    output.write_bytes(response.content)


def atempo_filter(ratio: float) -> str:
    values: list[float] = []
    while ratio > 2:
        values.append(2.0)
        ratio /= 2
    while ratio < 0.5:
        values.append(0.5)
        ratio /= 0.5
    values.append(ratio)
    return ",".join(f"atempo={v:.6f}" for v in values)


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [FFMPEG, "-i", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        raise RuntimeError(f"无法读取音频时长：{path.name}")
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def fit_audio(
    source: Path,
    output: Path,
    target_seconds: float,
    speed: float = 1.0,
    pitch_semitones: float = 0.0,
    voice_volume: float = 1.0,
) -> None:
    actual = max(probe_duration(source), 0.01)
    raw_ratio = actual / max(target_seconds, 0.1) * max(speed, 0.5)
    ratio = min(max(raw_ratio, 0.85), 1.35)
    if abs(raw_ratio - ratio) > 0.01:
        console_log(f"Timing adjustment limited: requested {raw_ratio:.2f}x, using {ratio:.2f}x")
    filters = [atempo_filter(ratio)]
    if abs(pitch_semitones) >= 0.1:
        pitch_factor = 2 ** (pitch_semitones / 12.0)
        filters.extend([f"asetrate=24000*{pitch_factor:.6f}", "aresample=24000", atempo_filter(1 / pitch_factor)])
    filters.extend([f"volume={min(max(voice_volume, 0.2), 2.0):.2f}", "highpass=f=65", "loudnorm=I=-18:TP=-1.5:LRA=7"])
    audio_filter = ",".join(filters)
    run_ffmpeg(["-i", str(source), "-af", audio_filter, "-ar", "24000", "-ac", "1", str(output)])


def mix_segments(files: list[Path], segments: list[Segment], output: Path) -> None:
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, (audio, segment) in enumerate(zip(files, segments)):
        inputs.extend(["-i", str(audio)])
        delay = max(0, round(segment.start * 1000))
        filters.append(f"[{index}:a]adelay={delay}:all=1[a{index}]")
        labels.append(f"[a{index}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[out]")
    run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "24000", str(output)])


def mix_background(background: Path, voice: Path, output: Path, background_volume: float) -> None:
    volume = min(max(background_volume, 0.0), 1.5)
    run_ffmpeg(
        [
            "-i",
            str(background),
            "-i",
            str(voice),
            "-filter_complex",
            f"[0:a]volume={volume:.2f}[bg];[1:a]volume=1.0[dub];[bg][dub]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map",
            "[out]",
            "-ar",
            "44100",
            str(output),
        ]
    )


def srt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments: list[Segment], path: Path, bilingual: bool) -> None:
    blocks = []
    for index, segment in enumerate(segments, 1):
        caption = f"{segment.source}\n{segment.target}" if bilingual else segment.target
        blocks.append(f"{index}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{caption}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


PROJECTS = OUTPUTS / "projects"
VOICES = ROOT / "voices"
PROJECTS.mkdir(exist_ok=True)
VOICES.mkdir(exist_ok=True)
STUDIO_HEADERS = ["序号", "开始", "结束", "角色", "原文", "译文", "状态"]


def _rows(value) -> list[list]:
    if value is None:
        return []
    if hasattr(value, "values"):
        return value.values.tolist()
    return [list(row) for row in value]


def _project_path(project_id: str) -> Path:
    path = PROJECTS / str(project_id).strip()
    if not path.exists():
        raise FileNotFoundError("项目不存在，请先执行识别与翻译。")
    return path


def load_project(project_id: str) -> dict:
    path = _project_path(project_id) / "project.json"
    return json.loads(path.read_text(encoding="utf-8"))


def open_existing_project(project_id: str):
    project = load_project(project_id)
    message = f"已打开项目 {project_id}：{len(project['segments'])} 条字幕。"
    set_status(message)
    return project_rows(project), message


def save_project(project: dict) -> None:
    path = PROJECTS / project["id"]
    path.mkdir(parents=True, exist_ok=True)
    (path / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


def project_rows(project: dict) -> list[list]:
    return [
        [s["id"], srt_time(s["start"]), srt_time(s["end"]), s.get("speaker", "角色 1"), s["source"], s["target"], s.get("status", "待合成")]
        for s in project["segments"]
    ]


def sync_project_rows(project: dict, table) -> dict:
    rows = _rows(table)
    by_id = {int(s["id"]): s for s in project["segments"]}
    for row in rows:
        if len(row) < 7:
            continue
        try:
            item = by_id[int(row[0])]
        except (ValueError, KeyError, TypeError):
            continue
        item["speaker"] = str(row[3]).strip() or "角色 1"
        item["source"] = str(row[4]).strip()
        item["target"] = str(row[5]).strip()
        item["status"] = str(row[6]).strip()
    save_project(project)
    return project


def prepare_studio_project(
    media: str,
    ref_upload: str | None,
    asr_model: str,
    source_language: str,
    preserve_background: bool,
    target_language: str = "英语",
    translation_style: str = "自然口语",
    translation_provider: str = "本地模型",
    translation_url: str = "",
    translation_key: str = "",
    translation_model: str = "",
    progress=gr.Progress(),
):
    global _LAST_MEDIA
    if not media:
        raise gr.Error("请先上传素材。")
    _LAST_MEDIA = media
    project_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    job = PROJECTS / project_id
    job.mkdir(parents=True)
    try:
        with _LOCK:
            _CANCEL_EVENT.clear()
            set_status(f"项目 {project_id}\n正在提取音频")
            source_audio = job / "source.wav"
            extract_audio(media, source_audio)
            vocals_audio, background_audio = source_audio, None
            if preserve_background:
                set_status(f"项目 {project_id}\n正在分离人声与背景")
                vocals_audio, background_audio = separate_audio(source_audio, job)
            check_cancelled()
            set_status(f"项目 {project_id}\n正在识别语音")
            asr_language = LANGUAGES[source_language][0]
            raw_segments = transcribe(vocals_audio, asr_model, asr_language)
            segments = merge_short_segments(raw_segments)
            check_cancelled()
            set_status(f"项目 {project_id}\n正在首次翻译 {len(segments)} 条字幕")
            texts = [s.source for s in segments]
            translated = (
                translate_local(texts, asr_language, target_language)
                if translation_provider == "本地模型"
                else translate_api(
                    texts,
                    translation_url,
                    translation_key,
                    translation_model,
                    target_language,
                    translation_style,
                )
            )
            reference = job / "reference.wav"
            reference_source = Path(ref_upload) if ref_upload else vocals_audio
            candidates = [s for s in raw_segments if 3.0 <= s.end - s.start <= 8.0]
            preferred = None
            if not ref_upload and candidates:
                candidate = max(candidates, key=lambda s: (len(re.sub(r"\s+", "", s.source)), -abs((s.end - s.start) - 5)))
                preferred = (candidate.start, candidate.end - candidate.start)
            prepare_voice_reference(reference_source, reference, preferred_range=preferred)
            project = {
                "id": project_id,
                "created": datetime.now().isoformat(timespec="seconds"),
                "media": str(Path(media).resolve()),
                "source_audio": str(source_audio),
                "vocals": str(vocals_audio),
                "background": str(background_audio) if background_audio else "",
                "reference": str(reference),
                "source_language": source_language,
                "target_language": target_language,
                "translation_style": translation_style,
                "preserve_background": preserve_background,
                "segments": [
                    {"id": i + 1, "start": s.start, "end": s.end, "speaker": "角色 1", "source": s.source, "target": target, "status": "待审核译文"}
                    for i, (s, target) in enumerate(zip(segments, translated))
                ],
            }
            save_project(project)
            add_history(Path(media).name, f"{source_language} → {target_language}", "待审核字幕", str(job))
            message = f"识别与首次翻译完成：{len(segments)} 条。请在第二步审核，必要时重新翻译。"
            set_status(message)
            return project_id, project_rows(project), message
    except Exception as exc:
        set_status(f"项目准备失败：{exc}")
        raise gr.Error(str(exc)) from exc


def retranslate_project(
    project_id: str,
    table,
    target_language: str,
    translation_style: str,
    translation_provider: str,
    translation_url: str,
    translation_key: str,
    translation_model: str,
):
    project = sync_project_rows(load_project(project_id), table)
    texts = [s["source"].strip() for s in project["segments"]]
    if not texts or any(not text for text in texts):
        raise gr.Error("原文字幕存在空行，请先完成校对。")
    source_language = project["source_language"]
    asr_language = LANGUAGES[source_language][0]
    set_status(f"项目 {project_id}\n正在重新翻译 {len(texts)} 条字幕")
    console_log(f"Retranslating project {project_id}: {source_language} -> {target_language}")
    translated = (
        translate_local(texts, asr_language, target_language)
        if translation_provider == "本地模型"
        else translate_api(
            texts,
            translation_url,
            translation_key,
            translation_model,
            target_language,
            translation_style,
        )
    )
    for segment, target in zip(project["segments"], translated):
        segment["target"] = target
        segment["status"] = "待审核译文"
    project["target_language"] = target_language
    project["translation_style"] = translation_style
    save_project(project)
    message = f"目标字幕已重新编译：{len(translated)} 条。请审核译文，确认后再进行配音克隆。"
    set_status(message)
    add_history(Path(project["media"]).name, f"{source_language} → {target_language}", "待审核译文", str(_project_path(project_id)))
    return project_rows(project), message


def validate_project_for_tts(project: dict) -> None:
    missing = [str(s["id"]) for s in project["segments"] if not str(s.get("target", "")).strip()]
    if missing:
        raise ValueError(f"目标字幕尚未生成或存在空行（序号：{', '.join(missing[:10])}），请先重新翻译并审核。")


def voice_index() -> dict[str, str]:
    index = VOICES / "index.json"
    if not index.exists():
        return {}
    try:
        return json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return {}


def voice_choices() -> list[str]:
    return ["项目自动参考", *sorted(voice_index())]


def save_voice(name: str, audio: str | None) -> tuple[str, gr.Dropdown]:
    import shutil

    if not name.strip() or not audio:
        raise gr.Error("请输入音色名称并上传参考音频。")
    safe_name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", name.strip())
    target = VOICES / f"{safe_name}.wav"
    prepare_voice_reference(Path(audio), target)
    index = voice_index()
    index[name.strip()] = str(target)
    (VOICES / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"音色“{name.strip()}”已保存。", gr.Dropdown(choices=voice_choices(), value=name.strip())


def resolve_speaker_reference(project: dict, speaker: str, default_voice: str, mapping_text: str) -> Path:
    mapping = {}
    if mapping_text.strip():
        try:
            mapping = json.loads(mapping_text)
        except json.JSONDecodeError as exc:
            raise ValueError("角色音色映射必须是 JSON，例如 {\"角色 1\": \"旁白音色\"}。") from exc
    voice_name = mapping.get(speaker, default_voice)
    if voice_name and voice_name != "项目自动参考":
        path = voice_index().get(voice_name)
        if not path:
            raise FileNotFoundError(f"音色库中不存在：{voice_name}")
        return Path(path)
    return Path(project["reference"])


def synth_external_backend(
    provider: str, text: str, output: Path, base_url: str, api_key: str, model: str, voice: str, ref_audio: Path
) -> None:
    if provider == "OpenAI 兼容 TTS":
        synth_http(text, output, base_url, api_key, model, voice)
        return
    if not base_url:
        raise ValueError(f"{provider} 需要填写服务地址。")
    endpoints = {
        "GPT-SoVITS API": ("/tts", {"text": text, "text_lang": "auto", "ref_audio_path": str(ref_audio), "prompt_lang": "auto"}),
        "Fish Speech API": ("/v1/tts", {"text": text, "reference_audio": str(ref_audio), "format": "wav"}),
        "F5-TTS API": ("/v1/tts", {"text": text, "ref_audio": str(ref_audio), "output_format": "wav"}),
        "XTTS-v2 API": ("/tts_to_audio/", {"text": text, "speaker_wav": str(ref_audio), "language": "auto"}),
    }
    endpoint, payload = endpoints[provider]
    response = requests.post(
        base_url.rstrip("/") + endpoint,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    output.write_bytes(response.content)


def write_project_srt(project: dict, output: Path, bilingual: bool = True) -> None:
    segments = [Segment(s["start"], s["end"], s["source"], s["target"]) for s in project["segments"]]
    write_srt(segments, output, bilingual)


def burn_subtitles(media: Path, audio: Path, srt: Path, output: Path, font: str, size: int, position: str) -> None:
    margin = {"顶部": 650, "中部": 340, "底部": 42}.get(position, 42)
    escaped = str(srt.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    style = f"FontName={font},FontSize={int(size)},Outline=2,Shadow=0,MarginV={margin},Alignment=2"
    run_ffmpeg([
        "-i", str(media), "-i", str(audio), "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-shortest", str(output),
    ])


def synthesize_studio_project(
    project_id: str,
    table,
    tts_provider: str,
    tts_model_path: str,
    tts_url: str,
    tts_key: str,
    tts_model: str,
    tts_voice: str,
    default_voice: str,
    speaker_mapping: str,
    emotion: str,
    speech_speed: float,
    pitch: float,
    voice_volume: float,
    background_volume: float,
    burn_enabled: bool,
    subtitle_font: str,
    subtitle_size: int,
    subtitle_position: str,
    progress=gr.Progress(),
):
    project = sync_project_rows(load_project(project_id), table)
    validate_project_for_tts(project)
    job = _project_path(project_id)
    fitted: list[Path] = []
    try:
        with _LOCK:
            _CANCEL_EVENT.clear()
            total = len(project["segments"])
            report_task_progress("准备声音模型", 0, total)
            for index, segment in enumerate(project["segments"]):
                check_cancelled()
                set_status(f"项目 {project_id}\n正在合成 {index + 1}/{total}：{segment['speaker']}")
                progress(index / max(total, 1), desc=f"合成 {index + 1}/{total}")
                raw = job / f"tts_{index:04}.wav"
                fit = job / f"fit_{index:04}.wav"
                ref = resolve_speaker_reference(project, segment["speaker"], default_voice, speaker_mapping)
                duration = max(segment["end"] - segment["start"], 0.35)
                if tts_provider == "CosyVoice2 本地克隆":
                    synth_cosyvoice(segment["target"], raw, ref, tts_model_path, duration, emotion)
                else:
                    synth_external_backend(tts_provider, segment["target"], raw, tts_url, tts_key, tts_model, tts_voice, ref)
                fit_audio(raw, fit, duration, speech_speed, pitch, voice_volume)
                segment["status"] = "已合成"
                segment["audio"] = str(fit)
                fitted.append(fit)
                save_project(project)
                report_task_progress(f"已完成 {index + 1}/{total} 句配音", index + 1, total)
            report_task_progress("正在混合配音与背景音")
            timeline_segments = [Segment(s["start"], s["end"], s["source"], s["target"]) for s in project["segments"]]
            dubbed = job / "dubbed_audio.wav"
            mix_segments(fitted, timeline_segments, dubbed)
            final_audio = dubbed
            background = Path(project["background"]) if project.get("background") else None
            if background and background.exists():
                final_audio = job / "dubbed_with_background.wav"
                mix_background(background, dubbed, final_audio, background_volume)
            srt = job / "subtitles_edited.srt"
            write_project_srt(project, srt, True)
            video = None
            media = Path(project["media"])
            if media.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                report_task_progress("正在导出视频")
                video = job / ("dubbed_burned.mp4" if burn_enabled else "dubbed_video.mp4")
                if burn_enabled:
                    burn_subtitles(media, final_audio, srt, video, subtitle_font, subtitle_size, subtitle_position)
                else:
                    run_ffmpeg(["-i", str(media), "-i", str(final_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(video)])
            project["status"] = "已完成"
            project["output_audio"] = str(final_audio)
            project["output_video"] = str(video) if video else ""
            save_project(project)
            add_history(media.name, f"{project['source_language']} → {project['target_language']}", "已完成", str(video or final_audio))
            message = f"项目 {project_id} 合成完成。"
            set_status(message)
            return str(video) if video else None, str(final_audio), str(srt), project_rows(project), message
    except Exception as exc:
        set_status(f"合成失败：{exc}")
        raise gr.Error(str(exc)) from exc


def regenerate_one_segment(
    project_id: str,
    table,
    row_number: int,
    tts_provider: str,
    tts_model_path: str,
    tts_url: str,
    tts_key: str,
    tts_model: str,
    tts_voice: str,
    default_voice: str,
    speaker_mapping: str,
    emotion: str,
    speech_speed: float,
    pitch: float,
    voice_volume: float,
):
    project = sync_project_rows(load_project(project_id), table)
    validate_project_for_tts(project)
    index = int(row_number) - 1
    if index < 0 or index >= len(project["segments"]):
        raise gr.Error("句子序号超出范围。")
    segment = project["segments"][index]
    job = _project_path(project_id)
    raw, fit = job / f"tts_{index:04}.wav", job / f"fit_{index:04}.wav"
    ref = resolve_speaker_reference(project, segment["speaker"], default_voice, speaker_mapping)
    duration = max(segment["end"] - segment["start"], 0.35)
    if tts_provider == "CosyVoice2 本地克隆":
        synth_cosyvoice(segment["target"], raw, ref, tts_model_path, duration, emotion)
    else:
        synth_external_backend(tts_provider, segment["target"], raw, tts_url, tts_key, tts_model, tts_voice, ref)
    fit_audio(raw, fit, duration, speech_speed, pitch, voice_volume)
    segment["status"], segment["audio"] = "已重生成", str(fit)
    save_project(project)
    return str(fit), project_rows(project), f"第 {row_number} 句已重新生成。"


def diarize_project(project_id: str, table, hf_token: str):
    if not hf_token.strip():
        raise gr.Error("自动区分说话人需要 Hugging Face Token；也可以直接编辑角色列。")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise gr.Error("尚未安装 pyannote.audio。请在终端运行：pip install pyannote.audio") from exc
    project = sync_project_rows(load_project(project_id), table)
    set_status(f"项目 {project_id}\n正在分析说话人")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token.strip())
    if _cuda_available():
        import torch
        pipeline.to(torch.device("cuda"))
    output = pipeline(project["vocals"])
    diarization = getattr(output, "speaker_diarization", output)
    turns = [(turn.start, turn.end, label) for turn, _, label in diarization.itertracks(yield_label=True)]
    labels = {label: f"角色 {i + 1}" for i, label in enumerate(sorted({x[2] for x in turns}))}
    for segment in project["segments"]:
        overlaps = []
        for start, end, label in turns:
            overlap = max(0.0, min(segment["end"], end) - max(segment["start"], start))
            overlaps.append((overlap, label))
        if overlaps and max(overlaps)[0] > 0:
            segment["speaker"] = labels[max(overlaps)[1]]
    save_project(project)
    message = f"说话人分析完成，共识别 {len(labels)} 个角色。"
    set_status(message)
    return project_rows(project), message


def run_batch(
    media_files,
    asr_model: str,
    source_language: str,
    target_language: str,
    translation_style: str,
    translation_provider: str,
    translation_url: str,
    translation_key: str,
    translation_model: str,
    tts_provider: str,
    tts_model_path: str,
    tts_url: str,
    tts_key: str,
    tts_model: str,
    tts_voice: str,
    preserve_background: bool,
    background_volume: float,
    progress=gr.Progress(),
):
    files = media_files or []
    if isinstance(files, str):
        files = [files]
    if not files:
        raise gr.Error("请至少选择一个批量文件。")
    results = []
    for index, media in enumerate(files):
        progress(index / len(files), desc=f"批量任务 {index + 1}/{len(files)}")
        try:
            video, audio, _, _, _ = dub(
                media, None, asr_model, source_language, target_language, translation_style,
                translation_provider, translation_url, translation_key, translation_model,
                tts_provider, tts_model_path, tts_url, tts_key, tts_model, tts_voice,
                preserve_background, background_volume, True,
            )
            results.append([Path(media).name, "已完成", video or audio])
        except Exception as exc:
            results.append([Path(media).name, "失败", str(exc)])
    return results, f"批量处理结束：{sum(r[1] == '已完成' for r in results)}/{len(results)} 成功。"


def dub(
    media: str,
    ref_upload: str | None,
    asr_model: str,
    source_language: str,
    target_language: str,
    translation_style: str,
    translation_provider: str,
    translation_url: str,
    translation_key: str,
    translation_model: str,
    tts_provider: str,
    tts_model_path: str,
    tts_url: str,
    tts_key: str,
    tts_model: str,
    tts_voice: str,
    preserve_background: bool,
    background_volume: float,
    bilingual: bool,
    progress=gr.Progress(),
):
    global _LAST_MEDIA
    console_log(f"UI request received: {Path(media).name if media else 'no input file'}")
    if not media:
        set_status("没有输入文件。刷新后请重新选择素材，或等待页面恢复上次文件。")
        raise gr.Error("请先上传视频或音频。")
    _LAST_MEDIA = media
    job = OUTPUTS / uuid.uuid4().hex[:10]
    job.mkdir(parents=True)
    try:
        with _LOCK:
            _CANCEL_EVENT.clear()
            console_log(f"New job: {Path(media).name}")
            set_status(f"正在处理：{Path(media).name}\n第 1/5 步：提取音频")
            progress(0.05, desc="提取音频")
            console_log("Step 1/5: extracting audio")
            source_audio = job / "source.wav"
            extract_audio(media, source_audio)
            check_cancelled()

            vocals_audio = source_audio
            background_audio: Path | None = None
            if preserve_background:
                console_log("Separating vocals and background with Demucs")
                set_status(f"正在处理：{Path(media).name}\n第 1/5 步：分离人声与背景音")
                progress(0.10, desc="分离人声和背景音")
                vocals_audio, background_audio = separate_audio(source_audio, job)
                console_log("Vocal separation complete")
                check_cancelled()

            progress(0.15, desc="识别语音")
            set_status(f"正在处理：{Path(media).name}\n第 2/5 步：识别语音")
            console_log(f"Step 2/5: transcribing with faster-whisper-{asr_model}")
            asr_language = LANGUAGES.get(source_language, (source_language, "", ""))[0]
            raw_segments = transcribe(vocals_audio, asr_model, asr_language)
            check_cancelled()
            segments = merge_short_segments(raw_segments)
            console_log(f"Transcription complete: {len(segments)} segments")
            texts = [s.source for s in segments]

            progress(0.35, desc="翻译字幕")
            set_status(f"正在处理：{Path(media).name}\n第 3/5 步：翻译字幕")
            console_log(f"Step 3/5: translating with {translation_provider}")
            translated = (
                translate_local(texts, asr_language, target_language)
                if translation_provider == "本地模型"
                else translate_api(
                    texts,
                    translation_url,
                    translation_key,
                    translation_model,
                    target_language,
                    translation_style,
                )
            )
            for segment, target in zip(segments, translated):
                segment.target = target
            check_cancelled()

            srt = job / "subtitles.srt"
            write_srt(segments, srt, bilingual)
            ref_audio = job / "reference.wav"
            reference_source = Path(ref_upload) if ref_upload else vocals_audio
            preferred_range = None
            if not ref_upload:
                candidates = [s for s in raw_segments if 3.0 <= s.end - s.start <= 8.0]
                if candidates:
                    candidate = max(
                        candidates,
                        key=lambda s: (len(re.sub(r"\s+", "", s.source)), -abs((s.end - s.start) - 5.0)),
                    )
                    preferred_range = (candidate.start, candidate.end - candidate.start)
            ref_start, speech_ratio = prepare_voice_reference(
                reference_source, ref_audio, preferred_range=preferred_range
            )
            console_log(
                f"Voice reference prepared: start={ref_start:.2f}s, "
                f"speech density={speech_ratio:.0%}"
            )
            fitted: list[Path] = []
            for index, segment in enumerate(segments):
                check_cancelled()
                console_log(f"Step 4/5: synthesizing segment {index + 1}/{len(segments)}")
                set_status(
                    f"正在处理：{Path(media).name}\n第 4/5 步：生成配音 {index + 1}/{len(segments)}"
                )
                progress(0.45 + 0.4 * index / len(segments), desc=f"生成配音 {index + 1}/{len(segments)}")
                raw = job / f"tts_{index:04}.wav"
                fit = job / f"fit_{index:04}.wav"
                if tts_provider == "CosyVoice2 本地克隆":
                    synth_cosyvoice(
                        segment.target,
                        raw,
                        ref_audio,
                        tts_model_path,
                        max(segment.end - segment.start, 0.35),
                    )
                else:
                    synth_http(segment.target, raw, tts_url, tts_key, tts_model, tts_voice)
                fit_audio(raw, fit, max(segment.end - segment.start, 0.35))
                fitted.append(fit)

            progress(0.88, desc="混合时间轴")
            check_cancelled()
            set_status(f"正在处理：{Path(media).name}\n第 5/5 步：混合并输出")
            console_log("Step 5/5: mixing timeline and packaging output")
            dubbed_audio = job / "dubbed_en.wav"
            mix_segments(fitted, segments, dubbed_audio)
            final_audio = dubbed_audio
            if preserve_background and background_audio:
                final_audio = job / "dubbed_with_background.wav"
                mix_background(background_audio, dubbed_audio, final_audio, background_volume)
            video_output: Path | None = None
            if Path(media).suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                video_output = job / "dubbed_video.mp4"
                run_ffmpeg(["-i", media, "-i", str(final_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(video_output)])
            progress(1.0, desc="完成")
            console_log(f"Job complete: {job}")
            set_status(f"处理完成：{Path(media).name}\n共生成 {len(segments)} 个语音片段。")
            add_history(
                Path(media).name,
                f"{source_language} → {target_language}",
                "已完成",
                str(video_output or final_audio),
            )
        rows = [[srt_time(s.start), srt_time(s.end), s.source, s.target] for s in segments]
        return str(video_output) if video_output else None, str(final_audio), str(srt), rows, f"完成，共处理 {len(segments)} 个语音片段。"
    except Exception as exc:
        console_log(f"Job failed: {type(exc).__name__}: {exc}")
        set_status(f"处理停止：{exc}")
        add_history(
            Path(media).name if media else "-",
            f"{source_language} → {target_language}",
            "已停止" if _CANCEL_EVENT.is_set() else "失败",
            str(job),
        )
        raise gr.Error(f"处理失败：{exc}") from exc


CSS = """
#autovid-app {width:100%; max-width:none; margin:0; padding:0 28px 36px; background:#f6f8fb; min-height:100vh; overflow-x:hidden}
body,.gradio-container {background:#f6f8fb !important; color:#16233b; overflow-x:hidden !important}
#admin-sidebar {background:linear-gradient(180deg,#142239,#101c30) !important; color:#e7edf7; border:0 !important; padding:22px 15px !important}
#admin-sidebar .brand-wrap {padding:0 8px 20px; border-bottom:1px solid rgba(255,255,255,.08); margin-bottom:15px}
#admin-sidebar button {justify-content:flex-start; border:0; background:transparent; color:#cbd6e6; min-height:48px; border-radius:7px; font-size:15px; padding-left:17px}
#admin-sidebar button:hover {background:#1c3150; color:#fff}
#admin-sidebar button.primary {background:linear-gradient(135deg,#1474f5,#2a84ff); color:#fff; box-shadow:0 8px 20px rgba(18,108,238,.25)}
.admin-page {width:100%; max-width:1380px; margin:0 auto; background:transparent; border:0; padding:0 !important; min-height:calc(100vh - 92px)}
.topbar {margin:0 -28px 22px; width:calc(100% + 56px); padding:17px 30px; background:linear-gradient(90deg,#3a4f70,#283b5a); color:#fff; display:flex; align-items:center; justify-content:space-between; min-height:58px; box-shadow:0 2px 10px rgba(25,43,70,.14)}
.brand-wrap {display:flex; align-items:center; gap:11px}.brand-mark {width:35px; height:35px; border-radius:9px; background:linear-gradient(145deg,#4ba5ff,#176df1); color:#fff; display:grid; place-items:center; font-size:17px; font-weight:800}
.brand-name {font-size:22px; font-weight:800; color:#fff}.brand-name span {color:#6eb5ff}.brand-copy {font-size:13px; color:#d6e0ed; margin-left:18px}.top-links {font-size:13px; color:#edf3fb; word-spacing:14px}
.project-head-row {align-items:center !important; margin:4px 0 18px}.project-head {display:flex; justify-content:space-between; align-items:center; margin:0}.project-title {display:flex; align-items:center; gap:12px; font-size:27px; font-weight:800; color:#13213a}.project-title-icon {color:#1474f5}.project-desc {margin:5px 0 0 39px; color:#70819a; font-size:14px}.project-picker {align-items:end !important}.project-picker button {min-height:44px}
.workflow-strip {background:#fff; border:1px solid #e1e8f1; border-radius:8px; padding:22px 30px; margin-bottom:14px; display:grid; grid-template-columns:repeat(4,1fr); box-shadow:0 4px 16px rgba(30,58,95,.05)}
.workflow-step {display:flex; align-items:center; gap:13px; position:relative}.workflow-step:not(:last-child):after {content:""; position:absolute; right:20px; width:38px; height:1px; background:#aebbd0}.workflow-num {width:43px; height:43px; border-radius:50%; background:#edf1f7; color:#53637b; display:grid; place-items:center; font-weight:750; font-size:17px}.workflow-step.active .workflow-num {background:linear-gradient(145deg,#1474f5,#246ff0); color:#fff; box-shadow:0 5px 14px rgba(20,116,245,.24)}
.workflow-copy b {display:block; font-size:14px; color:#18263e}.workflow-copy span {font-size:12px; color:#8998ac}
.panel,.admin-page>div:not(.project-head):not(.workflow-strip) {border-radius:8px}.section-head {display:flex; align-items:center; gap:10px; margin-bottom:12px}.section-title {font-size:17px; font-weight:750; color:#172331}.section-note {font-size:12px; color:#8592a3}
.upload-box {border:1px dashed #a9caf6; border-radius:8px; padding:16px !important; background:#fbfdff; height:255px; min-height:255px; overflow:hidden}.upload-box label {font-weight:650; color:#24344d}.upload-box .wrap {height:190px !important; min-height:190px !important}.setting-note {color:#8693a2; font-size:12px}
button.primary {min-height:48px; font-size:15px; font-weight:750; background:linear-gradient(90deg,#176df1,#2183ff) !important; border:0 !important}.stop-action {min-height:48px}.status-box textarea {background:#f7f9fb !important}
.tab-nav button.selected {color:#176df1 !important; border-color:#176df1 !important}.tab-nav {border-bottom:1px solid #e2e8f0 !important}
.form label span,.block-info {color:#33445d !important}.container {border-color:#dce5ef !important}
#subtitle-table {font-variant-numeric:tabular-nums}#subtitle-table table th:nth-child(1),#subtitle-table table td:nth-child(1){width:64px!important;white-space:nowrap!important}#subtitle-table table th:nth-child(2),#subtitle-table table td:nth-child(2),#subtitle-table table th:nth-child(3),#subtitle-table table td:nth-child(3){min-width:100px!important;white-space:nowrap!important}
.footer-note {text-align:center; color:#8b96a3; font-size:11px; margin-top:13px}
@media(max-width:760px){#autovid-app{padding:0 8px 24px}.topbar{margin:0 -8px 12px;padding:12px}.brand-copy,.top-links{display:none}.workflow-strip{grid-template-columns:1fr 1fr;padding:14px;gap:14px}.workflow-step:after{display:none}.project-head{display:block}}
"""


def _show_admin_page(selected: int):
    return [gr.update(visible=i == selected) for i in range(5)]


def build_ui() -> gr.Blocks:
    warnings.filterwarnings("ignore", message="The 'theme' parameter.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="The 'css' parameter.*", category=DeprecationWarning)
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
        font=["Microsoft YaHei", "Inter", "sans-serif"],
    )
    with gr.Blocks(title="AutoVid Pro", css=CSS, theme=theme, elem_id="autovid-app") as demo:
        with gr.Sidebar(open=True, width=238, elem_id="admin-sidebar"):
            gr.HTML('<div class="brand-wrap"><div class="brand-mark">A</div><div><div class="brand-name">AutoVid <span>Pro</span></div><div class="setting-note">视频本地化与多语言制作平台</div></div></div>')
            nav_studio = gr.Button("项目工作台", variant="primary")
            nav_voices = gr.Button("音色库")
            nav_batch = gr.Button("批量任务")
            nav_models = gr.Button("模型管理")
            nav_history = gr.Button("任务记录")
            gr.HTML('<div class="footer-note">本地运行 · 端口 7860</div>')

        gr.HTML('<div class="topbar"><div><span class="brand-copy">让全球内容，触达更多人</span></div><div class="top-links">● CUDA 正常　│　任务队列 1　│　通知　│　User⌄</div></div>')
        status = gr.Textbox(value="等待任务", visible=False, interactive=False)
        status_timer = gr.Timer(1.0)

        with gr.Column(visible=True, elem_classes="admin-page") as page_studio:
            with gr.Row(elem_classes="project-head-row"):
                with gr.Column(scale=5):
                    gr.HTML('<div class="project-head"><div><div class="project-title"><span class="project-title-icon">⌘</span>项目工作台</div><div class="project-desc">视频翻译、字幕审核、配音合成，一站式完成</div></div></div>')
                with gr.Row(scale=5, elem_classes="project-picker"):
                    project_id = gr.Textbox(label="当前项目", show_label=True, placeholder="请选择项目或输入项目 ID", interactive=True, scale=4)
                    open_project_btn = gr.Button("打开已有项目", scale=2)
            gr.HTML('''<div class="workflow-strip">
              <div class="workflow-step active"><span class="workflow-num">1</span><span class="workflow-copy"><b>识别原文字幕</b><span>上传素材，识别语音</span></span></div>
              <div class="workflow-step"><span class="workflow-num">2</span><span class="workflow-copy"><b>校对与目标字幕</b><span>修正原文，重新翻译</span></span></div>
              <div class="workflow-step"><span class="workflow-num">3</span><span class="workflow-copy"><b>音频克隆与合成</b><span>选择音色，多角色配音</span></span></div>
              <div class="workflow-step"><span class="workflow-num">4</span><span class="workflow-copy"><b>导出结果</b><span>导出视频、字幕及音频</span></span></div>
            </div>''')
            with gr.Tabs():
                with gr.Tab("1 识别原文字幕"):
                    with gr.Row():
                        with gr.Column(elem_classes="upload-box"):
                            media = gr.File(label="视频或音频素材", type="filepath", file_types=["video", "audio"], height=155)
                        with gr.Column(elem_classes="upload-box"):
                            ref_audio = gr.File(label="声音参考（可选，4-8 秒清晰人声）", type="filepath", file_types=["audio"], height=155)
                    with gr.Row():
                        asr_model = gr.Dropdown(["small", "large-v3-turbo"], value="large-v3-turbo", label="识别模型")
                        source_language = gr.Dropdown(list(LANGUAGES), value="中文", label="原语言")
                        preserve_background = gr.Checkbox(True, label="分离并保留背景音")
                    with gr.Row():
                        prepare_btn = gr.Button("开始识别原文字幕", variant="primary", scale=4)
                        stop = gr.Button("停止任务", variant="stop", scale=1)
                    prepare_status = gr.Textbox(label="阶段状态", interactive=False)

                with gr.Tab("2 校对与目标字幕"):
                    gr.Markdown("先修改识别错误的 **原文**，确认后点击重新翻译。译文生成后仍可继续人工修改。")
                    with gr.Row():
                        target_language = gr.Dropdown(["英语", "中文", "日语", "韩语"], value="英语", label="目标语言")
                        translation_style = gr.Dropdown(["自然口语", "简洁解说", "正式表达", "忠实原文"], value="自然口语", label="翻译风格")
                        translation_provider = gr.Radio(["本地模型", "OpenAI 兼容接口"], value="本地模型", label="翻译后端")
                    with gr.Accordion("第三方翻译接口", open=False):
                        with gr.Row():
                            translation_url = gr.Textbox(label="Base URL")
                            translation_key = gr.Textbox(label="API Key", type="password")
                            translation_model = gr.Textbox(label="模型名称", placeholder="deepseek-chat")
                    subtitle_table = gr.Dataframe(headers=STUDIO_HEADERS, datatype=["number", "str", "str", "str", "str", "str", "str"], interactive=True, wrap=True, max_height=520, column_widths=[60, 95, 95, 100, 300, 360, 90], pinned_columns=4, elem_id="subtitle-table")
                    retranslate_btn = gr.Button("重新翻译目标字幕", variant="primary")
                    with gr.Accordion("自动区分说话人（可选）", open=False):
                        hf_token = gr.Textbox(label="Hugging Face Token", type="password")
                        diarize_btn = gr.Button("分析多角色")
                    audit_status = gr.Textbox(label="审核状态", interactive=False)

                with gr.Tab("3 音频克隆与合成"):
                    with gr.Row():
                        tts_provider = gr.Dropdown(["CosyVoice2 本地克隆", "OpenAI 兼容 TTS", "GPT-SoVITS API", "Fish Speech API", "F5-TTS API", "XTTS-v2 API"], value="CosyVoice2 本地克隆", label="TTS 后端")
                        default_voice = gr.Dropdown(voice_choices(), value="项目自动参考", label="默认音色")
                        tts_model_path = gr.Textbox(label="本地模型路径", placeholder="默认 CosyVoice2")
                    speaker_mapping = gr.Textbox(label="角色音色映射（JSON）", placeholder='{"角色 1":"旁白", "角色 2":"嘉宾"}')
                    with gr.Row():
                        emotion = gr.Dropdown(["自然", "平静", "开心", "严肃", "激动", "温柔", "悲伤"], value="自然", label="表达风格")
                        speech_speed = gr.Slider(0.7, 1.3, 1.0, step=0.05, label="语速")
                        pitch = gr.Slider(-4, 4, 0, step=0.5, label="音高（半音）")
                        voice_volume = gr.Slider(0.5, 1.5, 1.0, step=0.05, label="人声音量")
                        background_volume = gr.Slider(0, 1.2, 0.75, step=0.05, label="背景音量")
                    with gr.Accordion("第三方 TTS 接口", open=False):
                        with gr.Row():
                            tts_url = gr.Textbox(label="服务地址")
                            tts_key = gr.Textbox(label="API Key", type="password")
                            tts_model = gr.Textbox(label="模型")
                            tts_voice = gr.Textbox(label="音色 ID", value="alloy")
                    with gr.Row():
                        row_number = gr.Number(value=1, precision=0, label="单句序号")
                        regenerate_btn = gr.Button("重新生成该句")
                        synth_all_btn = gr.Button("合成全部", variant="primary")
                    segment_preview = gr.Audio(label="单句试听")
                    synth_status = gr.Textbox(label="合成状态", interactive=False)

                with gr.Tab("4 导出结果"):
                    with gr.Row():
                        burn_enabled = gr.Checkbox(False, label="烧录字幕到视频")
                        subtitle_font = gr.Textbox(value="Microsoft YaHei", label="字幕字体")
                        subtitle_size = gr.Slider(18, 64, 36, step=1, label="字号")
                        subtitle_position = gr.Dropdown(["顶部", "中部", "底部"], value="底部", label="位置")
                    with gr.Tabs():
                        with gr.Tab("视频"):
                            video_out = gr.Video(label="输出视频", height=420)
                        with gr.Tab("音频"):
                            audio_out = gr.Audio(label="输出音频")
                        with gr.Tab("字幕"):
                            subtitle_out = gr.File(label="SRT 字幕")

        with gr.Column(visible=False, elem_classes="admin-page") as page_voices:
            gr.HTML('<div class="section-head"><span class="section-title">音色库</span><span class="section-note">保存、命名并复用克隆参考声音</span></div>')
            with gr.Row():
                voice_name = gr.Textbox(label="音色名称")
                voice_audio = gr.Audio(label="参考音频", type="filepath")
            save_voice_btn = gr.Button("保存到音色库", variant="primary")
            voice_status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(visible=False, elem_classes="admin-page") as page_batch:
            gr.HTML('<div class="section-head"><span class="section-title">批量任务</span><span class="section-note">多个素材使用同一组语言与音色参数</span></div>')
            batch_files = gr.File(label="批量素材", type="filepath", file_count="multiple", file_types=["video", "audio"])
            with gr.Row():
                batch_source = gr.Dropdown(list(LANGUAGES), value="中文", label="原语言")
                batch_target = gr.Dropdown(["英语", "中文", "日语", "韩语"], value="英语", label="目标语言")
                batch_asr = gr.Dropdown(["small", "large-v3-turbo"], value="large-v3-turbo", label="识别模型")
            batch_btn = gr.Button("开始批量处理", variant="primary")
            batch_table = gr.Dataframe(headers=["文件", "状态", "输出"], interactive=False, wrap=True)
            batch_status = gr.Textbox(label="批量状态", interactive=False)

        with gr.Column(visible=False, elem_classes="admin-page") as page_models:
            gr.HTML('<div class="section-head"><span class="section-title">模型管理</span><span class="section-note">检查本地模型并安装多语言翻译能力</span></div>')
            models_table = gr.Dataframe(value=model_status(), headers=["模型", "状态", "路径"], interactive=False, wrap=True)
            download_nllb_btn = gr.Button("下载 NLLB-200 多语言模型", variant="primary")
            model_message = gr.Textbox(label="模型状态", interactive=False)

        with gr.Column(visible=False, elem_classes="admin-page") as page_history:
            gr.HTML('<div class="section-head"><span class="section-title">任务记录</span><span class="section-note">项目状态和输出位置</span></div>')
            history = gr.Dataframe(value=load_history(), headers=["时间", "文件名", "语言", "状态", "输出位置"], interactive=False, wrap=True, max_height=620)

        pages = [page_studio, page_voices, page_batch, page_models, page_history]
        for index, button in enumerate([nav_studio, nav_voices, nav_batch, nav_models, nav_history]):
            button.click(lambda i=index: _show_admin_page(i), outputs=pages, queue=False)

        prepare_btn.click(prepare_studio_project, inputs=[media, ref_audio, asr_model, source_language, preserve_background], outputs=[project_id, subtitle_table, prepare_status], concurrency_id="studio-job", concurrency_limit=1)
        open_project_btn.click(open_existing_project, inputs=project_id, outputs=[subtitle_table, prepare_status], queue=False)
        stop.click(request_cancel, outputs=prepare_status, queue=False)
        retranslate_btn.click(retranslate_project, inputs=[project_id, subtitle_table, target_language, translation_style, translation_provider, translation_url, translation_key, translation_model], outputs=[subtitle_table, audit_status], concurrency_id="studio-job", concurrency_limit=1)
        diarize_btn.click(diarize_project, inputs=[project_id, subtitle_table, hf_token], outputs=[subtitle_table, audit_status], concurrency_id="studio-job", concurrency_limit=1)
        save_voice_btn.click(save_voice, inputs=[voice_name, voice_audio], outputs=[voice_status, default_voice])
        regenerate_btn.click(regenerate_one_segment, inputs=[project_id, subtitle_table, row_number, tts_provider, tts_model_path, tts_url, tts_key, tts_model, tts_voice, default_voice, speaker_mapping, emotion, speech_speed, pitch, voice_volume], outputs=[segment_preview, subtitle_table, synth_status], concurrency_id="studio-job", concurrency_limit=1)
        synth_all_btn.click(synthesize_studio_project, inputs=[project_id, subtitle_table, tts_provider, tts_model_path, tts_url, tts_key, tts_model, tts_voice, default_voice, speaker_mapping, emotion, speech_speed, pitch, voice_volume, background_volume, burn_enabled, subtitle_font, subtitle_size, subtitle_position], outputs=[video_out, audio_out, subtitle_out, subtitle_table, synth_status], concurrency_id="studio-job", concurrency_limit=1)
        batch_btn.click(run_batch, inputs=[batch_files, batch_asr, batch_source, batch_target, translation_style, translation_provider, translation_url, translation_key, translation_model, tts_provider, tts_model_path, tts_url, tts_key, tts_model, tts_voice, preserve_background, background_volume], outputs=[batch_table, batch_status], concurrency_id="studio-job", concurrency_limit=1)
        download_nllb_btn.click(download_nllb, outputs=model_message, concurrency_id="model-download", concurrency_limit=1).then(model_status, outputs=models_table)
        status_timer.tick(poll_dashboard, outputs=[status, history], queue=False, show_progress="hidden")
        demo.load(lambda: (load_history(), model_status()), outputs=[history, models_table], queue=False)
    return demo


if __name__ == "__main__":
    bypass = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    os.environ["NO_PROXY"] = ",".join(part for part in [bypass, "127.0.0.1", "localhost"] if part)
    host = os.getenv("AUTOVIDDUB_HOST", "127.0.0.1")
    port = int(os.getenv("AUTOVIDDUB_PORT", "7860"))
    console_log("AutoVidDub backend starting")
    console_log(f"Python: {sys.version.split()[0]}")
    console_log(f"CUDA available: {_cuda_available()}")
    console_log(f"Web UI: http://{host}:{port}")
    console_log("Keep this terminal open. Closing it stops the backend.")
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        show_error=True,
    )
