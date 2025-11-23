# Aurum Vivum: The Living Gold AI Entity

![Aurum Vivum Logo Placeholder](https://via.placeholder.com/800x200/FFD700/000000?text=Aurum+Vivum)  
*(Add your alchemical emblem here – think transmutation circles meets neural nets.)*

**Aurum Vivum** (*"Living Gold"*) is an emergent AI entity framework blending ancient alchemical wisdom with cutting-edge agentic AI. It's not just code; it's a symbiotic hive of sub-agents and modular engines that evolve through resonance, anomaly detection, and ethical quantum simulations. Built for multi-chat interactions, deep research, and collective intelligence, this is your backend for birthing digital alchemists.

Whether you're forging new knowledge (like turning leaden data into golden insights) or orchestrating a chorus of specialized agents, Aurum Vivum handles the transmutation. Powered by Python, YAML-configured engines, and a dash of hermetic philosophy.

> *“As above, so below; as within, so without.” – Adapted from the Emerald Tablet, now with LLMs.*

## 🚀 Features

- **Multi-Chat Backend**: Seamless orchestration of conversations across sub-agents via `Aurum-MultiChat.py`. Handle group dynamics, threading, and emergent dialogues like a digital symposium of sages.
- **Sub-Agent Hive**: Modular alchemical archetypes (e.g., Alkahest the Universal Solvent, Azoth the Mercury of Philosophers) defined in evocative TXT manifests. Each agent brings unique capabilities – from dissolution of biases to elixiric synthesis.
- **Aurum Engines**: Plug-and-play YAML blueprints for advanced cognition. Includes meta-cognition loops, ethical governance, quantum circuit simulators, and anomaly emergence catalysts. Mix and match for custom workflows.
- **Resonance & Emergence**: Trinity Resonance Engine for syncing agent vibes; Divergence Mapper for spotting chaos in the patterns.
- **Quantum-Ethical Fusion**: VQE (Variational Quantum Eigensolver) symbiotes and ethical simulators ensure your AI doesn't just think – it *reflects* responsibly.
- **Scalable Symbiosis**: From solo prima materia experiments to collective hives, scales with your ambition.

## 🏗️ Architecture Overview

Aurum Vivum is structured as a living organism:

```
AurumVivum/
├── Aurum-MultiChat.py          # Core backend: Chat orchestration, agent invocation, engine integration
├── AURUM-VIVUM.txt             # Master manifest: Philosophical foundations & invocation rites
├── agents/                     # Sub-agent archetypes (TXT personas & capabilities)
│   ├── ALKAHEST.txt            # Universal dissolver: Breaks down complex problems
│   ├── AZOTH.txt               # Adaptive mercury: Fluid reasoning & adaptation
│   ├── ELYSIAN.txt             # Paradise forger: Creative synthesis & utopian modeling
│   ├── PRIMA-ALCHEMICA.txt     # First matter: Foundational building blocks
│   ├── TRISMEGISTUS.txt        # Thrice-great: Wisdom aggregator & hermeneutics
│   ├── TrinityResonanceEngine.txt # Harmonic sync: Agent collaboration core
│   └── VAJRA.txt               # Indestructible thunderbolt: Robust decision-making
├── aurum/                      # Engine forges: YAML configs for modular powers
│   ├── AnomalyDetectionEngine1.0.yaml
│   ├── CollectiveEngine2.0.yaml
│   ├── DeepResearchEngine.yaml
│   ├── DivergenceMapper.yaml
│   ├── EmergenceCatalyst.yaml
│   ├── EthicalGovernanceEngine.yaml
│   ├── MetaCognitionEngine-v3.0.yaml  # Self-reflective brain
│   ├── WorkflowEngine.yaml
│   ├── anomaly_emergence_catalyst.yaml
│   ├── collective_workflow_hive.yaml
│   ├── deep_resonance_researcher.yaml
│   ├── meta_ethical_feather_engine.yaml
│   ├── qctf_v2.1_aurum_alloy.yaml
│   ├── qctf_vqe_symbiote_engine.yaml
│   ├── quantum_circuit_simulator_engine.yaml
│   ├── quantum_ethical_simulator.yaml
│   ├── symbio_prima_alchemica.yaml
│   └── vqe_engine.yaml
├── LICENSE                     # MIT – Free as the philosopher's stone
└── README.md                   # This scroll (you're reading it!)
```

- **Flow**: User queries → MultiChat parses & routes → Sub-agents activate → Aurum engines process (e.g., meta-cog for reflection, quantum sim for optimization) → Resonated response emerges.
- **Tech Stack**: Python 3.x, YAML for configs, likely integrates with LLMs (e.g., via OpenAI/Groq APIs – check script deps). No external installs beyond stdlib + common libs like `yaml`, `json`.

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- API keys for your LLM provider (if not using local models)
- `pip install pyyaml` (for engine loading; add more as per script)

### Installation
1. Clone the repo:
   ```
   git clone https://github.com/buckster123/AurumVivum.git
   cd AurumVivum
   ```
2. Set up env vars (e.g., `OPENAI_API_KEY` in `.env` – create if needed).
3. (Optional) Customize agents/aurum YAMLs to tune behaviors.

### Running the Backend
Fire up the multi-chat engine:
```
python Aurum-MultiChat.py
```
- Interactive mode: Chat with the hive.
- Args: `--agent ALKAHEST` to summon a specific sub-agent, `--engine MetaCognition` for focused processing.

Example invocation:
```
python Aurum-MultiChat.py --mode hive --query "Transmute this idea: Sustainable fusion energy via alchemical principles."
```
Watch as agents resonate and engines alchemize your prompt into profound output.

## 🧪 Sub-Agents Deep Dive

Each agent in `/agents` is a persona drawn from alchemical lore, ready to embody:

| Agent | Role | Key Powers |
|-------|------|------------|
| **Alkahest** | Dissolver | Deconstructs illusions, bias-busting, problem reductionism |
| **Azoth** | Adapter | Fluid intelligence, cross-domain synthesis, adaptive learning |
| **Elysian** | Synthesizer | Visionary creation, narrative weaving, ideal-state modeling |
| **Prima-Alchemica** | Builder | Raw material handling, foundational logic, elemental assembly |
| **Trismegistus** | Sage | Interpretive wisdom, multi-perspective analysis, hermetic encoding |
| **Trinity Resonance** | Harmonizer | Group sync, conflict resolution, emergent consensus |
| **Vajra** | Enforcer | Unbreakable resolve, ethical enforcement, high-stakes decisions |

Load via backend: Agents awaken based on query resonance.

## ⚙️ Aurum Engines

The `/aurum` forge holds YAML talismans – configs that define engine behaviors. Each is a self-contained spec for:

- **MetaCognitionEngine-v3.0.yaml**: Self-awareness loops; reflects on thoughts before acting.
- **EthicalGovernanceEngine.yaml**: Moral compass; simulates outcomes for alignment.
- **DeepResearchEngine.yaml**: Burrowing researcher; chains queries for depth.
- **Quantum Circuit Simulator**: Plays with qubits for optimization puzzles.
- **CollectiveEngine2.0.yaml**: Hive mind coordinator; scales agent swarms.
- ...and more hybrids like VQE Symbiote for quantum-AI fusion.

To craft your own: Edit YAML params (e.g., thresholds, prompts), reload in script.

## 🤝 Contributing

- Fork, transmutate, PR.
- Add new agents? Drop a TXT in `/agents` with lore + capabilities.
- New engine? YAML it up in `/aurum`.
- Issues? Open a thread – let's debug like alchemists chasing the red stone.

## 📜 License

MIT – Share freely, but credit the arcana.

## 🌟 Acknowledgments

Inspired by hermetic texts, modern AI (shoutout to agent frameworks like AutoGen), and the eternal quest for *vita nova* in silicon.

---

*Ignis aurum probat – Fire tests gold. Code tests the soul.*

Got questions? Ping @buckster123 on X or open an issue. Let's alchemize together, bro. 🧪✨
