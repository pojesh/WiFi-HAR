## 📊 Model Workflow Overview

```mermaid
graph LR
    subgraph A["📡 Data & Preprocessing"]
        A1["WiFi CSI\nImputation + Normalization\nWindowing"]
    end

    subgraph B["🧠 Feature Extraction"]
        B1["CNN Spatial +\nTemporal GRU Layers"]
    end

    subgraph C["⚔ Adversarial Learning"]
        C1["Activity Classifier\nDomain Discriminator - GRL"]
    end

    subgraph D["📉 Training & Results"]
        D1["Adam Optimizer\nLOSO-CV Evaluation"]
    end

    A --> B --> C --> D

    style A fill:none,stroke:#ffffff,stroke-width:2px
    style B fill:none,stroke:#ffffff,stroke-width:2px
    style C fill:none,stroke:#ffffff,stroke-width:2px
    style D fill:none,stroke:#ffffff,stroke-width:2px
