# 🖋️ PaleographIA: Transcription System for Jundiaí's Historical Archive

**PaleographIA** is a specialized high-performance HTR (Handwritten Text Recognition) system designed specifically for the historical documentation of Jundiaí (1657 - 1889). 

The system leverages state-of-the-art AI architecture (**TrOCR**) to decipher irregular caligraphy, paper textures from the 17th-19th centuries, and archaic Portuguese terminology.

![Project Status](https://img.shields.io/badge/STATUS-OPERACIONAL-green?style=for-the-badge)
![UI Mode](https://img.shields.io/badge/INTERFACE-SPECIALIST_MODE-purple?style=for-the-badge)
![Hardware](https://img.shields.io/badge/HARDWARE-CPU_SAFE_MODE-blue?style=for-the-badge)

## 🌟 Key Features

### 🧠 Deep Calibration for Portuguese Paleography
Unlike generic OCRs, PaleographIA uses a specialized fine-tuning process to understand the nuances of 300-year-old Italian paper texture and the specific cursive styles of regional scribes.
- **Literal Decoding (Greedy Search)**: Forced character-by-character interpretation to prevent English-biased hallucinations.
- **Noise Filtering**: Advanced CV2 filters to ignore vergê paper textures and ink stains.

### 💻 Hardware Stabilization (CPU Safe Mode)
Designed to run on notebook hardware (16GB RAM / GTX 1050 Ti) without causing system crashes or overheating.
- **Thread Limiting**: Controlled processing power to prevent TDR (Timeout Detection and Recovery) errors.
- **Cooling Pauses**: Wait times between line inferences to keep hardware temperatures stable.

### 🎨 Expert-First Interface
A distraction-free, high-contrast dark environment designed for long professional transcription sessions.
- **Intelligent Sidebar**: Real-time editor with horizontal/vertical manuscript sync.
- **Segmented Visualization**: Visual overlays on the document to track transcription progress line-by-line.

## 🛠️ Project Structure

```text
├── client/          # Next.js Frontend (Expert UI)
├── server/          # FastAPI Backend (AI Inference)
├── data/
│   ├── raw/         # Historical scans
│   └── processed/   # Segmented lines for training
├── models/          # Specialized TrOCR weights (TrOCR-Jundiahy)
└── scripts/         # Automated training & processing scripts
```

## 🚀 Getting Started

1. **Start the Backend**:
   Run the safe starter: `.\start_safe.bat`
   
2. **Start the Frontend**:
   ```bash
   cd client
   npm run dev
   ```

3. **Transcription**:
   Upload a scan, wait for the AI segmentation, and use the **Smart Editor** to finalize the transcription.

## 📜 History & Context
This project was developed to preserve and digitalize the history of Jundiaí, providing historians with a powerful assisted transcription stool that respects the complexity of original manuscripts while using modern AI to speed up the process.

---
*Developed for the Jundiaí Historical Archive (1615 - 2026).*
