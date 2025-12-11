# Neuromodulated-Tower-System

> **Biologically-Inspired Multi-Tower Architecture with Neuromodulator-Gated Integration**

A PyTorch-based implementation of a recursive self-improvement system inspired by Hansen et al.'s (2024) brainstem-cortex connectivity findings. This architecture implements 5 specialized processing towers with dynamic neurotransmitter-gated integration.

## Architecture Overview

### 5 Specialized Processing Towers

**Tower 1: Social-Memory**
- Autobiographical and episodic memory processing
- Social cognition and theory of mind
- Long-term pattern storage via EWC (Elastic Weight Consolidation)

**Tower 2: Working-Memory & Cognitive Control**
- Dynamic task management
- Cognitive state monitoring
- Meta-learning policy (learns when and how to use other towers)

**Tower 3: Affective Processing**
- 3-hormone neuromodulatory system (Dopamine, Serotonin, Cortisol)
- Emotional state representation
- Intrinsic motivation generation

**Tower 4: Sensorimotor Integration**
- Perception (vision, proprioception)
- Action decoding (dual-head: what & where/how)
- Sensory-motor binding

**Tower 5: Motor Coordination & Sequencing**
- Complex behavioral sequencing
- Planning and trajectory generation
- Execution of motor programs

### Neuromodulator Gating Layer

Based on PET imaging findings from Hansen et al. (2024):
- **18-receptor dynamic routing** system
- **3 main neurotransmitter pathways**: NET (norepinephrine), DAT (dopamine), 5-HTT (serotonin)
- **Context-dependent modulation**: Tower outputs weighted by current hormonal state and task context
- **Biologically-validated connectivity patterns**: Unimodal ↔ Transmodal hierarchy

### Recursive Self-Improvement Loop

1. **Parallel Tower Processing**: 5 towers process independently
2. **NT-Gated Integration**: Neuromodulator gate combines outputs
3. **Cortical Reasoning**: H-module style planning (inspired by Sapient HRM)
4. **Action Selection**: L-module execution with meta-learning
5. **Feedback Loop**: Gate weights updated via meta-cognition module

## Installation

```bash
git clone https://github.com/sunghunkwag/Neuromodulated-Tower-System.git
cd Neuromodulated-Tower-System
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from src.system import FiveTowerSystem

# Initialize system
system = FiveTowerSystem(
    latent_dim=128,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Process input state
state = torch.randn(1, 256)  # Batch size 1, 256-dim state
action, nt_weights = system(state)

print(f"Action shape: {action.shape}")
print(f"NT gate weights: {nt_weights}")
```

## Core Features

✅ **5-Tower Parallel Processing**: Specialized cognitive modules  
✅ **Neurotransmitter-Gated Integration**: Context-dependent routing  
✅ **Recursive Meta-Learning**: Self-improving gate weights  
✅ **Biologically-Plausible**: Grounded in neuroscience (Hansen et al., 2024)  
✅ **PyTorch Native**: Full GPU support and autograd compatibility  
✅ **Modular Design**: Each tower independently trainable  

## Project Structure

```
Neuromodulated-Tower-System/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── towers/
│   │   ├── __init__.py
│   │   ├── tower_base.py
│   │   ├── tower1_social_memory.py
│   │   ├── tower2_working_memory.py
│   │   ├── tower3_affective.py
│   │   ├── tower4_sensorimotor.py
│   │   └── tower5_motor_coordination.py
│   ├── neuromodulator_gate.py
│   ├── system.py
│   └── training.py
├── tests/
│   └── test_system.py
└── notebooks/
    └── demo.ipynb
```

## Key Papers & References

1. **Hansen et al. (2024)** - "Brainstem-Cortex Connectivity and Hierarchical Cognition"  
   *Nature Neuroscience* - Found 5-community structure in brainstem nuclei  

2. **Sutskever (2024)** - Digital Brainstem Concept  
   *SSI Research Direction*  

3. **Sapient (2024)** - HRM (Hierarchical Reasoning Model)  
   *ArXiv* - H-module + L-module dual processing  

4. **Kahneman (2011)** - Thinking Fast and Slow  
   *Dual-Process Theory Foundation*  

## Training & Evaluation

### Basic Training Loop

```python
from src.training import train_epoch
from src.system import FiveTowerSystem
import torch
from torch.optim import Adam

system = FiveTowerSystem(latent_dim=128, device='cuda')
optimizer = Adam(system.parameters(), lr=1e-3)

for epoch in range(100):
    loss = train_epoch(system, optimizer, train_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}")
```

### Testing

```bash
python -m pytest tests/test_system.py -v
```

## Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this in research, please cite:

```bibtex
@software{kwag2025neuromodulated,
  author = {Kwag, Sunghun},
  title = {Neuromodulated-Tower-System: A Biologically-Inspired Multi-Tower Architecture},
  year = {2025},
  url = {https://github.com/sunghunkwag/Neuromodulated-Tower-System}
}
```

## Author

Sunghun Kwag - Independent AI Research  
GitHub: [@sunghunkwag](https://github.com/sunghunkwag)

---

**Status**: 🚀 Active Development  
**Last Updated**: December 2025
