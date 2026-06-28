# Crowd Surge Risk Prediction System


Real-time stampede risk detection using a novel dual-stream
spatio-temporal deep learning architecture on the HAJJv2 dataset.
Autonomous alerting with zero human intervention required.

---

## The Problem

Mass gatherings like Hajj (2+ million people) pose extreme stampede
risk. Existing systems rely on static cameras and manual monitoring
-- too slow to prevent casualties. This system predicts surge risk
before it becomes a stampede, in real time.

---

## Architecture

Video Stream

|

v

+------------------------------------------+

|           SPATIAL STREAM                 |

|  YOLOv8 -> ByteTrack -> SWIN Transformer |

|  (person detection + scene features)     |

+------------------+-----------------------+

|

v

+------------------------------------------+

|           TEMPORAL STREAM                |

|         Bi-LSTM Classifier               |

|  (sequential movement pattern learning)  |

+------------------+-----------------------+

|

v

+------------------------------------------+

|         ALPHA FUSION LAYER               |

|  Novel risk scoring module               |

|  Inputs: density + velocity + variance   |

|  Output: alpha risk score (0.0 to 1.0)  |

+------------------+-----------------------+

|

+--------+--------+

v                 v

Risk Classification      Automated Alert

4 severity tiers         Twilio SMS

SAFE / LOW /             fired instantly

MEDIUM / HIGH            at threshold

|

v

Streamlit Dashboard

Live heatmaps + zone risk + alert history

---

## Dataset - HAJJv2

Source       : Real-world Hajj crowd footage
Scale        : 100,000+ annotated frames
Classes      : Normal, Light Panic, Violent Group, Fight, Stampede
Challenge    : High-density, high-acceleration surge videos

---

## Results

False alert reduction : 30% via threshold calibration
Risk classes          : 4 severity tiers
Alert latency         : Real-time per video window
Validation            : Peer-reviewed, IEEE Access & Springer 2025

---

## Key Innovations

- Novel AlphaLayer: custom fusion module combining spatial density
  change rate, speed variance, and temporal risk with Gaussian
  smoothing into a single interpretable risk score

- Surge-specific design: built for high-acceleration crowd movement,
  not static crowd counting

- Zero human intervention: fully autonomous detection to
  classification to alert pipeline

- Published research: validated through peer review in two
  indexed journals

---

## Tech Stack

Detection      : YOLOv8
Tracking       : ByteTrack
Spatial model  : SWIN Transformer
Temporal model : Bi-LSTM
Fusion         : Custom AlphaLayer
Backend        : FastAPI
Dashboard      : Streamlit
Alerting       : Twilio SMS API
Language       : Python

---

## Setup

git clone https://github.com/Ramkumar64/crowd-surge-risk-prediction.git
cd crowd-surge-risk-prediction
pip install -r requirements.txt
python fusion/alpha_fusion_main.py

Note: Model weights (.pt files) are excluded due to size (110MB each).
Contact for access or retrain using the provided training scripts.

---

## Repository Structure

fusion/
    alpha_fusion_main.py        - Main fusion pipeline
    detection_tracking.py       - YOLOv8 + ByteTrack integration
    train_risk_predictor.py     - AlphaLayer training
train_window_swin.py            - SWIN Transformer training
train_temporal_classifier.py    - Bi-LSTM training
run_infer.py                    - Inference pipeline
visualize_video.py              - Output visualization
assets/                         - System diagrams and sample outputs

---

## Publication

Integrated Spatio-Temporal Deep Learning Framework for
Crowd Behaviour Analysis
- IEEE Access, 2025
- Springer, 2025

---

## Author

Ramkumar R
AI & Data Science, Madras Institute of Technology, Anna University (2026)
Email   : ramaravind21135@gmail.com
LinkedIn: linkedin.com/in/ram-kumar-r-01951636a
GitHub  : github.com/Ramkumar64
