from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from faster_whisper.audio import decode_audio

# Adjust path so modules can be imported
sys.path.insert(0, str(Path(__file__).parents[1]))
from lumacaption.gpu_runtime import configure_cuda_runtime
from lumacaption.mt.nllb_engine import NllbEngine
from lumacaption.stt.whisper_engine import WhisperEngine

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def edit_distance(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        new_d = [i + 1] * (len(hyp) + 1)
        for j, h in enumerate(hyp):
            cost = 0 if r == h else 1
            new_d[j + 1] = min(d[j + 1] + 1, new_d[j] + 1, d[j] + cost)
        d = new_d
    return d[-1]

def compute_wer(ref: str, hyp: str) -> float:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)

def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(normalize_text(ref).replace(" ", ""))
    hyp_chars = list(normalize_text(hyp).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)

def load_samples(manifest_path: Path, split: str | None = None) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    samples = []
    for item in data["samples"]:
        if split and item["split"] != split:
            continue
        audio_path = root / item["path"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing sample audio: {audio_path}")
        raw = audio_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != item["sha256"]:
            raise ValueError(f"Checksum mismatch for {item['path']}: {actual_sha} vs {item['sha256']}")
        samples.append({
            **item,
            "full_path": audio_path,
        })
    return samples

def run_benchmark(
    samples: list[dict],
    models: list[str],
    beams: list[int],
    device: str = "cuda",
    test_mt: bool = True,
) -> dict:
    configure_cuda_runtime()
    app_root = Path(__file__).parents[1]
    whisper_cache = app_root / "models" / "whisper"
    nllb_cache = app_root / "models" / "nllb-cache"

    # Preload and convert all audio to int16 mono PCM
    audio_pcm = {}
    for s in samples:
        flt = decode_audio(str(s["full_path"]), sampling_rate=16000)
        pcm = np.clip(flt * 32767.0, -32768, 32767).astype(np.int16)
        audio_pcm[s["id"]] = (pcm, len(pcm) / 16000.0)

    results = {}

    for model_name in models:
        for beam in beams:
            key = f"{model_name}-beam{beam}"
            print(f"\n--- Menguji {key} pada {device.upper()} ---", flush=True)
            engine = WhisperEngine(model_name, device, whisper_cache, beam_size=beam)
            engine.prepare("id")
            print(f"Engine siap: {engine.active_model} / {engine.active_device}", flush=True)

            sample_results = []
            total_wer = 0.0
            total_cer = 0.0
            total_stt_time = 0.0
            total_audio_duration = 0.0

            for s in samples:
                pcm, duration = audio_pcm[s["id"]]
                t0 = time.monotonic()
                transcript = engine.transcribe(pcm, "id")
                stt_duration = time.monotonic() - t0

                wer = compute_wer(s["text"], transcript.text)
                cer = compute_cer(s["text"], transcript.text)

                total_wer += wer
                total_cer += cer
                total_stt_time += stt_duration
                total_audio_duration += duration

                sample_results.append({
                    "id": s["id"],
                    "duration_s": round(duration, 2),
                    "ref": s["text"],
                    "hyp": transcript.text,
                    "wer": round(wer, 4),
                    "cer": round(cer, 4),
                    "stt_ms": round(stt_duration * 1000, 1),
                })

            engine.close()

            n = len(samples)
            avg_wer = total_wer / n
            avg_cer = total_cer / n
            avg_stt_ms = (total_stt_time / n) * 1000
            rtf = total_stt_time / total_audio_duration

            print(f"Hasil {key}: WER={avg_wer*100:.2f}%, CER={avg_cer*100:.2f}%, "
                  f"Avg STT={avg_stt_ms:.1f}ms, RTF={rtf:.3f}", flush=True)

            results[key] = {
                "model": model_name,
                "beam": beam,
                "device": device,
                "samples_count": n,
                "avg_wer": round(avg_wer, 4),
                "avg_cer": round(avg_cer, 4),
                "avg_stt_ms": round(avg_stt_ms, 1),
                "rtf": round(rtf, 4),
                "details": sample_results,
            }

    if test_mt:
        print(f"\n--- Menguji NLLB Machine Translation pada {device.upper()} ---", flush=True)
        mt = NllbEngine("mijuanlo/nllb-200-distilled-600M-ct2-int8", device, nllb_cache)
        mt.prepare(["English", "Japanese"])
        print(f"MT engine siap: {mt.active_device}", flush=True)

        mt_times = []
        mt_samples = []
        for s in samples:
            t0 = time.monotonic()
            trans = mt.translate(s["text"], "ind_Latn", ["English", "Japanese"])
            mt_duration = (time.monotonic() - t0) * 1000
            mt_times.append(mt_duration)
            mt_samples.append({
                "id": s["id"],
                "ref_id": s["text"],
                "en": trans.get("English", ""),
                "ja": trans.get("Japanese", ""),
                "mt_ms": round(mt_duration, 1),
            })

        mt.close()
        avg_mt = sum(mt_times) / len(mt_times)
        p95_mt = sorted(mt_times)[int(len(mt_times) * 0.95)]
        print(f"Hasil NLLB MT: Rata-rata={avg_mt:.1f}ms, P95={p95_mt:.1f}ms", flush=True)
        results["mt_nllb"] = {
            "device": device,
            "avg_mt_ms": round(avg_mt, 1),
            "p95_mt_ms": round(p95_mt, 1),
            "details": mt_samples,
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark caption STT and MT")
    parser.add_argument("--split", choices=["tuning", "evaluation", "all"], default="tuning")
    parser.add_argument("--models", nargs="+", default=["medium", "large-v3-turbo"])
    parser.add_argument("--beams", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = Path(r"C:\Users\Kenewjr\.gemini\antigravity-ide\brain\4d99ab57-0e07-467c-9ff4-fa109ed8455b\scratch\fleurs_id\manifest.json")
    split_filter = None if args.split == "all" else args.split
    samples = load_samples(manifest, split=split_filter)
    print(f"Loaded {len(samples)} samples (split: {args.split}) from FLEURS manifest.")

    res = run_benchmark(samples, args.models, args.beams, device=args.device)

    out_path = Path(args.out) if args.out else Path(r"C:\Users\Kenewjr\.gemini\antigravity-ide\brain\4d99ab57-0e07-467c-9ff4-fa109ed8455b\scratch") / f"benchmark_{args.split}.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHasil disimpan ke: {out_path}")
