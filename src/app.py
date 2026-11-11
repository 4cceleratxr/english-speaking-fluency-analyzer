"""
English Speaking Fluency Analyzer
Automated Speaking Rate, Pause Segmentation, Filler-Word Frequency, and LLM Coaching
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def load_audio_mono_16k(audio_path):
    """
    Loads audio file and converts it to 16 kHz mono float32 array.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return y, sr

def detect_pauses_and_phonation(y, sr=16000, frame_length_ms=16, energy_threshold=0.01):
    """
    Detects acoustic pause intervals and computes phonation ratio.
    Classifies pauses into Micro-Pauses (0.15-0.25s), Natural Breath Pauses (0.25-0.60s),
    and Hesitation Pauses (>0.60s).
    """
    hop_length = int(sr * (frame_length_ms / 1000.0))
    win_length = hop_length * 2
    rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)[0]
    
    is_speech = rms > energy_threshold
    duration = len(y) / sr
    
    pauses = []
    current_pause_start = None
    
    for i, active in enumerate(is_speech):
        t = i * (frame_length_ms / 1000.0)
        if not active:
            if current_pause_start is None:
                current_pause_start = t
        else:
            if current_pause_start is not None:
                pause_dur = t - current_pause_start
                if pause_dur >= 0.15:
                    if pause_dur <= 0.25:
                        p_type = "micro"
                    elif pause_dur <= 0.60:
                        p_type = "natural"
                    else:
                        p_type = "hesitation"
                    pauses.append({"start": current_pause_start, "end": t, "duration": pause_dur, "type": p_type})
                current_pause_start = None
                
    if current_pause_start is not None:
        pause_dur = duration - current_pause_start
        if pause_dur >= 0.15:
            p_type = "micro" if pause_dur <= 0.25 else ("natural" if pause_dur <= 0.60 else "hesitation")
            pauses.append({"start": current_pause_start, "end": duration, "duration": pause_dur, "type": p_type})
            
    total_pause_time = sum(p["duration"] for p in pauses)
    phonation_time = max(0.0, duration - total_pause_time)
    phonation_ratio = (phonation_time / duration) * 100.0 if duration > 0 else 0.0
    
    return {
        "duration": duration,
        "pauses": pauses,
        "phonation_ratio": phonation_ratio,
        "natural_pauses": sum(1 for p in pauses if p["type"] == "natural"),
        "hesitation_pauses": sum(1 for p in pauses if p["type"] == "hesitation")
    }

def analyze_disfluencies(transcript, duration=60.0):
    """
    Detects vocalized fillers (um, uh, er, ah) and discourse crutches (like, you know, basically).
    """
    filler_words = {"um", "uh", "er", "ah", "like", "you know", "basically", "actually"}
    words = transcript.lower().split()
    total_words = len(words)
    
    fillers_found = [w for w in words if w in filler_words]
    filler_count = len(fillers_found)
    filler_density = (filler_count / total_words * 100.0) if total_words > 0 else 0.0
    
    speaking_rate = (total_words / duration) * 60.0
    
    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_density": filler_density,
        "speaking_rate_wpm": speaking_rate
    }

def compute_composite_fluency_score(speaking_rate_wpm, phonation_ratio, hesitation_pauses, filler_density):
    """
    Computes 0-100 composite fluency score mapped to CEFR bands.
    Pace Optimality (30 pts), Pause & Phonation (35 pts), Disfluency Control (35 pts).
    """
    pace_diff = abs(speaking_rate_wpm - 150.0)
    pace_score = max(0.0, 30.0 - (pace_diff * 0.4))
    
    phonation_score = min(20.0, (phonation_ratio / 80.0) * 20.0)
    hesitation_penalty = min(15.0, hesitation_pauses * 1.5)
    pause_score = max(0.0, 35.0 - (15.0 - phonation_score + hesitation_penalty))
    
    disfluency_score = max(0.0, 35.0 - (filler_density * 2.5))
    
    total_score = min(100.0, max(0.0, pace_score + pause_score + disfluency_score))
    
    if total_score >= 85:
        cefr = "C1/C2 (Advanced)"
    elif total_score >= 65:
        cefr = "B1/B2 (Intermediate)"
    else:
        cefr = "A2 (Developing)"
        
    return {
        "pace_score": round(pace_score, 1),
        "pause_score": round(pause_score, 1),
        "disfluency_score": round(disfluency_score, 1),
        "composite_score": round(total_score, 1),
        "cefr_band": cefr
    }

def main():
    print("English Speaking Fluency Analyzer CLI")
    print("Experiential Learning Project - KIIT")

if __name__ == "__main__":
    main()
