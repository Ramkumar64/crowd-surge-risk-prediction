\# Crowd Surge Risk Prediction System

\### Published: IEEE Access \& Springer, 2025



A real-time crowd surge risk prediction platform that detects 

stampede-risk behaviour using a novel dual-stream deep learning 

architecture — deployed with automated alerting and live monitoring.



\## Architecture

\- \*\*Spatial stream:\*\* YOLOv8 + SWIN Transformer for scene-level 

&#x20; crowd density and behaviour classification

\- \*\*Temporal stream:\*\* Bi-LSTM for sequential movement pattern analysis

\- \*\*AlphaLayer fusion:\*\* Novel risk scoring module combining both 

&#x20; streams to compute a 0–1 risk score across 4 severity tiers

\- \*\*Alerting:\*\* Automated Twilio SMS alerts on high-risk detection

\- \*\*Dashboard:\*\* Streamlit real-time monitoring with heatmaps and 

&#x20; zone-level risk display



\## Dataset

HAJJv2 — real-world Hajj crowd footage, 100K+ annotated frames,  

5 behaviour classes: Normal, Light Panic, Violent Group, Fight, Stampede



\## Results

\- 30% reduction in false alert rate through threshold calibration

\- Published and peer-reviewed in IEEE Access and Springer (2025)



\## Tech Stack

Python · YOLOv8 · ByteTrack · SWIN Transformer · Bi-LSTM · 

FastAPI · Streamlit · Twilio



\## Setup

```bash

pip install -r requirements.txt

python fusion/alpha\_fusion\_main.py

```



\## Publication

Research cited in IEEE Access and Springer, 2025.

