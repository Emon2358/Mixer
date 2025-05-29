#!/usr/bin/env python3
# mix.py — Advanced Auto-Mix Script (MP3 & M4A 対応)

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment, effects
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ----- Configuration -----
CLIP_LENGTH_MS      = 36 * 1000      # 36秒
TARGET_LUFS         = -14.0          # ラウドネス正規化値
MIN_CROSSFADE_MS    = 300            # フェード最小幅
MAX_CROSSFADE_MS    = 1500           # フェード最大幅
N_CLUSTERS          = None           # None=曲数/10 or min 2
OUTPUT_FILE         = "mixed_auto.mp3"

def analyze_track(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    # BPM検出
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
    # オンセット位置を取得（バックトラック有効）
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    start_frame = onsets[0] if len(onsets) else 0
    start_ms = int(start_frame * (512/sr) * 1000)
    # ハーモニック特徴量：クロマ＋Tonnetz
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1)
    feat = np.concatenate(([tempo], chroma, tonnetz))
    # ラウドネス
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    return {
        'path':      path,
        'tempo':     tempo,
        'sr':        sr,
        'start_ms':  start_ms,
        'loudness':  loudness,
        'feat':      feat
    }

def load_and_analyze():
    # MP3 と M4A を両対応で読み込む
    paths = sorted(glob.glob("*.mp3") + glob.glob("*.m4a"))
    with ThreadPoolExecutor() as exe:
        return list(tqdm(exe.map(analyze_track, paths), total=len(paths), desc="Analyzing"))

def cluster_tracks(tracks):
    n = len(tracks)
    k = max(2, n//10) if N_CLUSTERS is None else N_CLUSTERS
    data = np.stack([t['feat'] for t in tracks])
    km = KMeans(n_clusters=k, random_state=0).fit(data)
    for i, t in enumerate(tracks):
        t['cluster'] = int(km.labels_[i])
    return tracks

def order_tracks(tracks):
    feats = np.stack([t['feat'] for t in tracks])
    nbrs = NearestNeighbors(n_neighbors=len(tracks), metric='cosine').fit(feats)
    used = set()
    order = [tracks[0]]
    used.add(tracks[0]['path'])
    current = feats[0].reshape(1, -1)
    while len(order) < len(tracks):
        dists, idxs = nbrs.kneighbors(current, n_neighbors=len(tracks))
        for dist, idx in zip(dists[0], idxs[0]):
            cand = tracks[idx]
            if cand['path'] not in used:
                order.append(cand)
                used.add(cand['path'])
                current = cand['feat'].reshape(1, -1)
                break
    return order

def make_clip(t, target_tempo=None):
    seg = AudioSegment.from_file(t['path'])
    clip = seg[t['start_ms']: t['start_ms'] + CLIP_LENGTH_MS]
    # Tempo stretch（次トラックのテンポに合わせる）
    if target_tempo:
        rate = target_tempo / t['tempo']
        y, sr = librosa.load(t['path'], sr=None, mono=True)
        start = int(t['start_ms']/1000*sr)
        end   = start + int(CLIP_LENGTH_MS/1000*sr)
        y_seg = y[start:end]
        y_ts  = librosa.effects.time_stretch(y_seg, rate)
        clip = AudioSegment(
            y_ts.tobytes(), frame_rate=int(sr*rate),
            sample_width=clip.sample_width, channels=1
        )
    # ラウドネス正規化（RMSベース + LUFS）
    clip = effects.normalize(clip)
    arr = np.array(clip.get_array_of_samples()).astype(np.float32)
    clip = pyln.normalize.loudness(arr, t['loudness'], TARGET_LUFS)
    return clip

def main():
    tracks = load_and_analyze()
    tracks = cluster_tracks(tracks)
    ordered = order_tracks(tracks)

    output = AudioSegment.silent(duration=0)
    for i, t in enumerate(ordered):
        next_tempo = ordered[i+1]['tempo'] if i+1 < len(ordered) else None
        clip = make_clip(t, target_tempo=next_tempo)
        fade = int(np.interp(
            abs((next_tempo or t['tempo'])-t['tempo']),
            [0, 20], [MAX_CROSSFADE_MS, MIN_CROSSFADE_MS]
        ))
        output += clip.fade_in(fade).fade_out(fade)

    output.export(OUTPUT_FILE, format="mp3")
    print(f"[Done] Created ▶ {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
