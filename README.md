# Franka AI

https://github.com/huggingface/lerobot/blob/790d6740babad57398e20a956a23068d377640f0/examples/3_train_policy.py

## 1. Project structure

### 1.1 Folder structure

```bash
franka_AI_/
│
├── README.md
├── environment.yml               # Conda environment (opzionale se vuoi anche pip)
│
├── configs/
│   ├── train_vla.yaml
│   ├── eval_bc.yaml
│   └── model_arch.yaml           # Configurazioni per modelli e dataset
│
├── data/
│   ├── lerobot_datasets/
│   │   └── softbag_pickplace/    # (symlink o path ai dati reali)
│   └── processed/                # Versioni preprocessate
│
├── src/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── lerobot_loader.py     # Wrapper per caricare dataset lerobot
│   │   └── transforms.py         # Preprocessing, augmentations, ecc.
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── policy_bc.py          # Behaviour cloning baseline
│   │   ├── policy_vla.py         # Vision-Language-Action architecture
│   │   ├── encoders/
│   │   │   ├── clip_encoder.py
│   │   │   ├── mae_encoder.py
│   │   │   └── proprio_encoder.py
│   │   └── decoders/
│   │       └── action_decoder.py
│   │
│   ├── training/
│   │   ├── train.py              # Main training loop
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── utils.py
│   │
│   ├── inference/
│   │   ├── run_policy.py         # Script per testare in simulazione o real
│   │   └── visualizations.py
│   │
│   ├── control/
│   │   ├── franka_interface.py   # API verso robot reale o simulator (es. pybullet, ROS2)
│   │   └── grasping.py
│   │
│   ├── utils/
│       ├── logger.py             # WandB, tensorboard, ecc.
        ├── seed.py
        └── config.py             # Parser YAML → dict
```

### 1.2 Setup conda environment and run the code

Clone repo and go inside folder:

```bash
git clone https://github.com/Argo-Robot/controls
cd controls
```

Download necessary libs by using conda environment:

```
conda env create -f environment.yml
conda activate controls_env
pip install --upgrade --force-reinstall networkx==2.8.8
```

To run `main_dh.py` or `main_urdf.py` just type:

```bash
cd controls
python main_dh.py
python main_urdf.py
```

To run `main_so101_mj.py` just type:

```bash
cd controls
python -m simulator.main_so101_mj
```

---

## 6. Known issues

---

## 7. References

- **P.I. Corke (2017)**: [*A Robotics Toolbox for MATLAB*](https://petercorke.com/toolboxes/robotics-toolbox/)

---

## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/Argo-Robot/kinematics/blob/main/LICENSE) file for details.
