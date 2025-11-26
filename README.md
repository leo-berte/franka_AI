# Franka AI

https://github.com/huggingface/lerobot/blob/790d6740babad57398e20a956a23068d377640f0/examples/3_train_policy.py

## 1. Project structure

### 1.1 Folder structure

```bash
franka_ai/
├── configs/
├── dataset/
├── outputs/
├── src/
│   └── franka_ai/
│       ├── dataset/
│       ├── training/
│       ├── inference/
│       ├── models/
│       ├── utils/
│       └── tests/
├── ros2/
│   └── franka_ai_inference/
│       ├── package.xml
│       ├── CMakeLists.txt
│       ├── launch/
│       └── src/inference_node.py
├── Dockerfile.train
├── Dockerfile.ros
└── README.md
```



docker build -t franka_ai -f Dockerfile.train .
sh docker_run_train.sh


https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
python -c "import torch; print(torch.cuda.is_available())"










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
