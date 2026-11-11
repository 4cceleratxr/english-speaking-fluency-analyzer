# English Speaking Fluency Analyzer

**Experiential Learning Project — Language Processing Using Machine Learning and LLMs for Audio, Image, Text, and Other Applications**  
**Institution:** Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar, Odisha  
**Submission Deadline:** 12 November 2026  
**Group Members:** Aman Kumar Srivastava (`2330287`) & Shivansh Giri (`2330114`)

## 1. Project Overview
The **English Speaking Fluency Analyzer** is an end-to-end audio and natural language processing application that evaluates a student's recorded **one-minute English speech**. The system combines acoustic Voice Activity Detection (VAD), disfluency-preserving Automatic Speech Recognition (ASR) with word-level timestamps, quantitative prosodic feature extraction, and an instruction-tuned Large Language Model (LLM) to compute objective fluency metrics and deliver actionable coaching feedback.

### Core Capabilities
- **Speaking Rate & Articulation Rate Calculation:** Computes overall **Words Per Minute (WPM)** across the full 60-second recording, **Articulation Rate (WPM)** during active phonation (excluding silence), and rolling **10-second window pacing trajectories** to detect rushing or fatigue.
- **Acoustic Pause Segmentation & Taxonomy:** Uses Silero VAD and short-time RMS energy thresholding (`16 ms` hop resolution at `16 kHz`) to classify silent intervals into **Micro-Pauses** (`0.15–0.25 s`), **Natural Breath Pauses** (`0.25–0.60 s`), and **Hesitation Pauses** (`>0.60 s`).
- **Filler-Word & Disfluency Detection:** Conditions OpenAI Whisper (`openai/whisper-small`) via Faster-Whisper (`Systran/faster-whisper-large-v3`) with a verbatim disfluency prompt so vocalized fillers (`um`, `uh`, `er`, `ah`) and discourse crutches (`like`, `you know`, `basically`, `actually`) are retained and timestamped rather than stripped.
- **Composite Fluency Scoring & CEFR Mapping:** Aggregates **Pace Optimality (`30 pts`)**, **Pause & Phonation Control (`35 pts`)**, and **Disfluency Control (`35 pts`)** into a `0–100` Composite Fluency Score mapped to **CEFR Spoken Fluency Bands (`A2` to `C1/C2`)**.
- **Automated LLM Coaching Feedback:** Passes the structured acoustic-lexical JSON profile to `Qwen/Qwen2.5-7B-Instruct` (or `meta-llama/Llama-3.1-8B-Instruct`) to synthesize timestamp-grounded pedagogical recommendations.

## 2. Repository Directory Structure
- `README.md` — Project documentation & execution guide
- `requirements.txt` — Python package dependencies
- `src/app.py` — Main CLI + Gradio Web UI application
- `notebooks/English_Speaking_Fluency_Analyzer.ipynb` — Interactive pipeline walkthrough & plots
- `sample_inputs/sample_student_speech_60s.wav` — Representative 60s 16kHz mono speech audio
- `sample_outputs/` — Benchmark evaluation JSON & CSV metrics for Samples A, B, and C
- `assets/` — RMS energy envelope, VAD pauses, 10s WPM, and subscore charts
- `docs/` — Project Report (`English_Speaking_Fluency_Analyzer_Report.pdf`) and Presentation Slides (`English_Speaking_Fluency_Analyzer_Presentation.pptx`)

## 3. Sample Evaluation Results (60-Second Benchmark Profiles)
| Sample ID | Speaker Profile | Speaking Rate (WPM) | Articulation Rate (WPM) | Phonation Ratio (%) | Natural Pauses | Hesitation Pauses | Filler Count (Density) | Composite Score | CEFR Band |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S01** | Intermediate Learner (Hesitant) | `108.0` | `156.5` | `69.0%` | `6` | `8` | `14` (`13.0%`) | **`43.2 / 100`** | `A2 (Developing)` |
| **S02** | Proficient Speaker (Balanced) | `146.0` | `178.0` | `82.0%` | `6` | `1` | `3` (`2.1%`) | **`94.5 / 100`** | `C1/C2 (Advanced)` |
| **S03** | Rapid Speaker (Rushed) | `184.0` | `206.7` | `89.0%` | `2` | `2` | `11` (`6.0%`) | **`70.4 / 100`** | `B1 (Intermediate)` |

## 4. Group Members and Individual Contributions
| Member Name | Roll / Registration No. | Role & Technical Contributions |
| :--- | :--- | :--- |
| **Aman Kumar Srivastava** | `2330287` | **Project Lead, Audio DSP & Whisper ASR Pipeline Architect:** Designed the 5-stage system architecture, implemented 16 kHz audio normalization, RMS/Silero VAD pause segmentation, OpenAI Whisper word-timestamp alignment (`src/app.py`), and `Qwen/Qwen2.5-7B-Instruct` coaching prompt synthesis. |
| **Shivansh Giri** | `2330114` | **NLP Disfluency Scoring, Gradio Web UI & Evaluation Lead:** Built the vocalized/discourse filler-word and repetition detection engine, 10-second rolling WPM tracker, `0–100` CEFR composite scoring rubric, interactive Gradio web dashboard, Jupyter notebook, and multi-profile benchmark evaluation. | 1-Minute Student Speech (KIIT Experiential Learning Project)
