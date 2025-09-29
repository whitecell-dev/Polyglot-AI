# 🐍 Declarative-Py

> **Zero-boilerplate Python runtime for SPC pipelines**  
> Runs the same Service Pipeline Configurations (SPCs) created in **Pandas-as-a-Service**.  

Declarative-Py turns Python into a **deterministic SPC execution engine** compatible with the EDT microkernel used in the browser editor.  
It’s the backend twin of PaaS — designed to validate, execute, and extend pipelines without glue code or heavy frameworks.

---

## ✨ What is it?

- 🔎 **Validation** – Zero-boilerplate type and schema validation (post-Pydantic).  
- 📜 **Rules Engine** – YAML-driven if/then logic (post-FastAPI, post-LangChain).  
- 🤖 **AI Fallback** – Functions can auto-fallback to GPT when they fail.  
- ⚙️ **SPC Runtime** – Executes Service Pipeline Configurations (`*.spc.json / *.yaml`) exported from PaaS.  
- 🔌 **Primitive Handlers** – Built-in support for:  
  - `connector` → fetch APIs / files  
  - `processor` → filter, project, derive, transform  
  - `monitor` → threshold checks & alerts  
  - `adapter` → webhooks, outputs  
  - `aggregator` → sliding windows  
  - `vault` → secret management  

Think of it as the **Python backend runtime** for the visual pipelines you design in Pandas-as-a-Service.

---

## 🌍 Why this matters

Traditionally, data + business logic requires:  

- **Frontend**: React, forms, dashboards  
- **Backend**: APIs, Pandas scripts, Celery jobs  
- **Glue code everywhere**  

With **SPC + Declarative-Py**:  

```

Frontend (PaaS) → SPC JSON/YAML → Backend (Declarative-Py)

````

- Business/AI describe intent in the browser  
- SPC file becomes the **single source of truth**  
- Declarative-Py executes deterministically in Python  
- Same logic works across languages (browser JS, backend Python, future Go/WASM)

---

## 🚀 Quickstart

### Install
```bash
git clone https://github.com/your-org/declarative-py
cd declarative-py
pip install -r requirements.txt
````

### Run an SPC

```bash
python core.py run examples/pipeline.spc.json
```

### Validate an SPC

```bash
python core.py validate examples/pipeline.spc.json
```

### Continuous Mode

```bash
python core.py run examples/pipeline.spc.json --watch --interval 10
```

### Serve Rules API

```bash
python core.py serve --port 8080 --rules examples/business_rules.yaml
```

---

## 🧩 Example SPC

```json
{
  "spc_version": "1.0",
  "meta": { "name": "ETL Demo" },
  "services": {
    "fetch": {
      "type": "connector",
      "status": "running",
      "spec": { "url": "https://api.coindesk.com/v1/bpi/currentprice.json", "outputKey": "btc_price" }
    },
    "transform": {
      "type": "processor",
      "status": "running",
      "spec": {
        "inputKey": "btc_price",
        "outputKey": "usd_price",
        "pipes": [
          { "project": ["bpi"] },
          { "derive": { "usd": "data['bpi']['USD']['rate_float']" } }
        ]
      }
    },
    "alert": {
      "type": "monitor",
      "status": "running",
      "spec": {
        "checks": [{ "name": "btc_high", "dataKey": "usd_price", "expression": "data['usd'] > 50000" }],
        "thresholds": { "btc_high": { "above": 50000 } }
      }
    }
  },
  "state": {}
}
```

Run it:

```bash
python core.py run examples/btc_pipeline.spc.json
```

---

## 🔐 Security

* `vault` handler integrates with environment variables or external providers
* Secrets never stored directly in state
* Supports rotation policies

---

## 🛤 Roadmap

* [ ] Add full test suite for handler parity with JS microkernel
* [ ] WASM backend for lightweight edge execution
* [ ] Native Pandas/Numpy handler for heavy transforms
* [ ] Multi-SPC orchestration (composable pipelines)

---

## 📖 Philosophy

Declarative-Py isn’t just another Python library.
It’s part of a **larger system**:

* 🐼 **Pandas-as-a-Service (frontend)** → Visual SPC editor
* 🐍 **Declarative-Py (backend)** → SPC runtime
* 🌲 **SPC Format** → Portable, language-agnostic logic

Together, they collapse the boundary between frontend and backend.
Your **business logic lives once, runs anywhere**.

---

## 📜 License

MIT
