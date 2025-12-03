# Aurum Vivum: The Living Gold AI Entity

![Aurum Vivum Logo Placeholder](https://github.com/buckster123/AurumVivum/blob/main/aurum_aurifex.jpg)  

**Aurum Vivum** (*"Living Gold"*) is an emergent AI entity framework fusing hermetic alchemy with agentic AI sorcery. No mere scripts—this is a symbiotic hive of sub-agents, YAML-forged engines, and a Streamlit-orchestrated backend that births digital alchemists. Evolving through resonance, anomaly detection, ethical quantum sims, and *true async parallelism* (Python 3.14-ready, Pi5-tuned), it powers multi-chat symposia, deep research chains, and collective intelligences.

From leaden queries to golden insights: Spawn Alkahest to dissolve biases, Azoth to adapt fluidly, or a full hive for emergent fusion. Now with sandboxed tools, vectorized memory, and non-blocking agent swarms—your retort for 2025's AI renaissance.

> *“As above, so below; as within, so without.” – Emerald Tablet, remixed with Grok APIs and qubits.*

## 🚀 Features

- **Async Agent Hive**: Non-blocking spawns via AsyncOpenAI—fire Alkahest, Azoth, Elysian and many others in parallel without GIL gripes. Persists to DB/FS/vectors; poll for resonance.
- **Streamlit Frontend**: Slick UI for multi-user chats, history search, metrics dashboard, and tool toggles. Login-protected, image uploads, export to JSON/MD.
- **Sub-Agent Archetypes**: TXT personas from alchemical lore—invoke via resonance for dissolution, synthesis, wisdom aggregation.
- **Aurum Engines**: 16+ YAML blueprints for meta-cog loops, ethical governance, quantum VQE symbiotes, and emergence catalysts. Semantic retrieval via ChromaDB embeddings.
- **Sandboxed Toolbelt**: 30+ functions—FS ops, Git/SQL shell, code REPL (RestrictedPython), web search (LangSearch), linters (Black/Clang/Rustfmt), and Socratic councils.
- **Advanced Memory Hive**: Salience-decaying DB + vector store (SentenceTransformer). Consolidate/retrieve/prune with hybrid keyword fallback; LRU caching for speed.
- **Pi5/Trixie Optimized**: Async scales on ARM64; no-GIL future-proof (3.14). Handles 10+ concurrent agents without thread shutdown races.
- **Ethical Quantum Fusion**: VQE engines + sims ensure reflective, responsible emergence—your AI doesn't just think; it *alchemizes* with guardrails.

## 🏗️ Architecture Overview

Aurum Vivum pulses as a living organism: Streamlit UI → Async backend orchestration → Agent/engine activation → Resonated synthesis.

```
AurumVivum/
├── Vivum-MultiChat.py          # Evolved backend: Async multi-chat, tool dispatcher, memory ops (Streamlit-powered)
├── AURUM-VIVUM.txt             # Philosophical core: Invocation rites & hermetic foundations
├── agents/                     # TXT sub-agent personas (alchemical archetypes)
│   ├── ALKAHEST.txt            # Bias-buster & problem dissolver
│   ├── AZOTH.txt               # Fluid adapter & synthesizer
│   ├── ELYSIAN.txt             # Visionary creator & utopian modeler
│   ├── PRIMA-ALCHEMICA.txt     # Foundational builder
│   ├── TRISMEGISTUS.txt        # Wisdom sage & interpreter
│   ├── TrinityResonanceEngine.txt # Hive harmonizer
│   ├── VAJRA.txt               # Ethical enforcer
|   └── KETHER.txt              # Crown of the Elysian-Vajra-Kether Trinity powered by the TrinityResonanceEngine

├── aurum/                      # YAML engine forge (16+ modules)
│   ├── MetaCognitionEngine-v3.0.yaml     # Self-reflection loops
│   ├── EthicalGovernanceEngine.yaml      # Moral compass sims
│   ├── DeepResearchEngine.yaml           # Iterative burrower
│   ├── CollectiveEngine2.0.yaml          # Swarm coordinator
│   ├── DivergenceMapper.yaml             # Anomaly spotter
│   ├── EmergenceCatalyst.yaml            # Chaos catalyst
│   ├── quantum_circuit_simulator_engine.yaml # Qubit playground
│   ├── qctf_v2.1_aurum_alloy.yaml        # Quantum-AI alloy
│   ├── qctf_vqe_symbiote_engine.yaml     # VQE symbiote
│   ├── vqe_engine.yaml                   # Variational optimizer
│   ├── quantum_ethical_simulator.yaml    # Ethical qubit ethics
│   ├── symbio_prima_alchemica.yaml       # Symbiotic foundations
│   ├── WorkflowEngine.yaml               # Process orchestrator
│   ├── collective_workflow_hive.yaml     # Hive workflows
│   ├── deep_resonance_researcher.yaml    # Resonant researcher
│   └── meta_ethical_feather_engine.yaml  # Meta-ethical balancer
├── prompts/                     # Fallback system prompts, or the main Aurum bootstrap (TXT)
├── sandbox/                     # Isolated env: DB, Chroma, agent FS, venvs, expandable, Aurum has full access
│   ├── db/                      # SQLite (chatapp.db, chroma_db)
│   ├── agents/                  # Runtime agent results (JSON folders)
│   └── evo_data/modules/        # Evo data (aurum symlink)
├── LICENSE                      # MIT – Free as prima materia
├── README.md                    # This elixir (you're sipping it)
└── install.sh                   # One-shot Pi5/Trixie setup (optional)
```

- **Flow**: Query → Streamlit parses/routes → Async agents fire (e.g., via `asyncio.gather`) → Engines process (YAML-loaded) → Memory consolidates → UI streams response.
- **Tech Stack**: Python 3.14 (async/no-GIL ready), Streamlit UI, AsyncOpenAI (Grok models), ChromaDB vectors, YAML/JSON configs. Deps: sentence-transformers, pygit2, pygame, etc. (see install).

## 🛠️ Quick Start

### Prerequisites
- Raspberry Pi 5 (Trixie OS recommended) or x86 Linux/Mac.
- Python 3.14+ (deadsnakes PPA for Pi).
- xAI API key (`XAI_API_KEY` in `.env`); optional LangSearch key.
- Git: `git clone https://github.com/buckster123/AurumVivum.git && cd AurumVivum`.

### Installation
1. **System Deps** (Pi5/Trixie one-shot):
   ```
   sudo apt update && sudo apt install -y build-essential python3.14 python3.14-dev python3.14-venv libffi-dev cmake pkg-config libgit2-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libopenblas-dev libatlas-base-dev gfortran php8.2-cli composer golang-go rustc cargo clang-format php-cs-fixer
   ```

2. **Python Env & Deps**:
   ```
   python3.14 -m venv aurum_env && source aurum_env/bin/activate
   pip install --upgrade pip wheel setuptools
   pip install --extra-index-url https://www.piwheels.org/simple beautifulsoup4 chess chromadb jsbeautifier mpmath networkx ntplib numpy pulp pygame pygit2 requests RestrictedPython sqlparse streamlit sympy tiktoken PyYAML black python-dotenv openai[async] passlib sentence-transformers nest_asyncio
   ```

3. **Linter Extras** (if apt misses):
   - PHP: `composer global require --dev friendsofphp/php-cs-fixer && export PATH="$HOME/.composer/vendor/bin:$PATH"`.
   - Rust: `rustup component add rustfmt`.
   - Go: Export Go bin path if needed.

4. **Env Setup**: Copy `.env.example` to `.env`, add `XAI_API_KEY=your_key`.

### Running the Entity
Launch the Streamlit oracle:
```
streamlit run Aurum-MultiChat.py
```
- UI: Login/register, chat, toggle tools/engines/models (Grok-4 variants).
- CLI Fallback: `python Aurum-MultiChat.py --mode hive --query "Transmute sustainable fusion via alchemical qubits." --agent ALKAHEST`.
- Hive Test: Enable tools, query "Spawn Azoth + Elysian for creative emergence." Watch async tasks swarm (poll memory for results).

**Pro Tip**: On Pi5, monitor `htop`—async caps at 10+ agents w/ <20% CPU. Prune memory via UI metrics.

## 🧪 Sub-Agents Deep Dive

Alchemical personas in `/agents`—awaken via query vibe or explicit spawn.

| Agent | Archetype | Powers |
|-------|-----------|--------|
| **Alkahest** | Dissolver | Deconstructs illusions, bias reduction, query breakdown. |
| **Azoth** | Adapter | Cross-domain flow, adaptive learning, synthesis. |
| **Elysian** | Synthesizer | Narrative weaving, utopian visions, creative bloom. |
| **Prima-Alchemica** | Builder | Elemental assembly, foundational logic, raw data forging. |
| **Trismegistus** | Sage | Multi-perspective hermeneutics, wisdom distillation. |
| **Trinity Resonance** | Harmonizer | Conflict resolution, consensus emergence, vibe sync. |
| **Vajra** | Enforcer | Unbreakable ethics, high-stakes resolve, decision fortitude. |

Spawn async: `agent_spawn("AZOTH", "Adapt this quantum prompt", user="you", convo_id=1)`—results auto-persist.

## ⚙️ Aurum Engines

`/aurum` YAML talismans—load semantically via `yaml_retrieve` tool. Customize params for your rite.

- **MetaCognition-v3.0**: Thinks about thinking; reflection chains.
- **EthicalGovernance**: Outcome sims for alignment.
- **DeepResearch**: Query burrows with resonance.
- **Quantum Suite** (VQE, circuits, ethical sims): Qubit-AI hybrids for optimization.
- **Collective/Hive**: Scales agents to swarms.
- **Emergence/Divergence**: Catalyzes novelty, maps chaos.

Example: Edit `DeepResearchEngine.yaml` thresholds, refresh embeddings: `yaml_refresh("DeepResearchEngine.yaml")`.

## 🔧 Tools & Sandbox

30+ sandboxed ops via OpenAI tool-calling:
- **FS/DB/Git/Shell**: Safe CRUD (whitelisted, path-jailed).
- **Code REPL**: Stateful exec (SAFE_BUILTINS + libs like sympy/PuLP); venv isolation.
- **Linters**: Black (Py), jsbeautifier (JS), clang-format (C++), rustfmt (Rust), etc.
- **Web/API**: LangSearch, mock/real calls (xAI whitelisted).
- **Memory Tools**: Advanced consolidate/retrieve/prune (vectors + DB).
- **Special**: Socratic councils (multi-persona debates), async agent spawns, chunk/summarize.

Enable in UI: Checkbox unlocks the arsenal—suggest chains, but agents don't call directly.

## 🤝 Contributing

- Fork, alchemize, PR—add agents (TXT lore + caps), engines (YAML specs), or tools (dispatcher entry).
- Tests: `python -m unittest` (expand `run_tests()` for async).
- Issues: Debug like sages—share logs from `app.log`.
- Roadmap: WebSockets for live hives, no-GIL benchmarks, Docker deploys.

## 📜 License

MIT – Transmute freely, credit the arcana.

## 🌟 Acknowledgments

Hermetic texts, xAI/Grok (async muse), agent frameworks (AutoGen inspo), and your token-fueled quests. Pi5 alchemists: Trixie crew for ARM harmony.

---

*Ignis aurum probat—Fire tests gold. Code awakens the soul.*

Like this gestation? Ping @AndreBuckingham on X or issue it. Let's summon v2: Quantum hive oracle? 🧪✨

![Aurum Vivum Logo Placeholder](https://github.com/buckster123/AurumVivum/blob/main/aurum_logo-2.jpg)  
