# Franka AI

End-to-end **AI pipeline for robotic manipulation** (covering dataset post-processing, training and real-time inference) built around fully containerized Docker environments. The project integrates state-of-the-art policies (ACT, Diffusion Policies) through the LeRobot framework, as well as custom flow-matching–based models using action-chunking and hybrid state-vision inputs. It is designed for **real-world logistics applications**, specifically pick-and-place of deformable soft bags using a Franka Panda robot.

**Key Features**

- Integrated baseline models: ACT and Diffusion Policy via LeRobot 
- Developed custom models: Transformer Flow-Matching
- Custom data transforms: multiple pose representations (quaternion / axis-angle / 6D), absolute or relative EE poses, optional past-action conditioning
- Image augmentation & state noise injection during training
- Training optimizations: AMP, AdamW, linear + cosine LR scheduling
- YAML-driven configuration for transforms, training and models parameters (no code changes required)
- ROS2 real-time inference designed for multi-rate sensors and safe deployment on Franka Panda
- Fully Dockerized setup for training and inference

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
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.inference
└── README.md
```

### 1.2 Setup docker for training

Clone repo and go inside folder:

```bash
git clone https://github.com/leo-berte/franka_ai
cd franka_ai
```

Build docker image:

```bash
docker build -t franka_ai_train -f docker/Dockerfile.train .
```

Enter in the container by accessing the new *franka_ai_train* image:

```bash
sh docker/docker_run_train.sh
```

### 1.3 Setup docker for inference

Clone repo and go inside folder:

```bash
git clone https://github.com/leo-berte/franka_ai
cd franka_ai
```

Build docker image:

```bash
docker build -t franka_ai_inference -f docker/Dockerfile.inference .
```

Enter in the container by accessing the new *franka_ai_inference* image:

```bash
sh docker/docker_run_inference.sh
```

### 1.4 Docker requirements

To use GPU on docker, you need to install on your machine the NVIDIA Container Toolkit, by following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Once the process is finished, enter a docker container (*sh docker_run_train.sh*) and verify installation with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Run training scripts

Once inside the container, you can execute the different scripts.

### 2.1 Test dataloader

Creates the train/validation split for the selected dataset and verifies that samples can be loaded correctly through the dataloader after applying custom transforms.

```bash
python src/franka_ai/tests/test_dataloader.py --dataset /workspace/data/demo1 --policy diffusion --config config1
```

### 2.2 Generate transformed dataset stats

Computes new dataset statistics after applying the custom transforms. The policy will automatically use the generated file *episode_stats_transformed.jsonl*.

```bash
python src/franka_ai/dataset/generate_updated_stats.py --dataset /workspace/data/demo1 --config config1
```

### 2.3 Train

Starts the training for the selected policy using the specified dataset path. Optionally, you can provide a directory containing pretrained checkpoints.

```bash
python src/franka_ai/training/train.py --dataset /workspace/data/demo1 \
                                       --pretrained /workspace/outputs/checkpoints/demo1_xxx/model.pt \
                                       --policy diffusion \
                                       --config config1
```

### 2.4 Evaluate

Starts the training for the selected policy using the specified dataset path. Optionally, you can provide a directory containing pretrained checkpoints.

```bash
python src/franka_ai/inference/evaluate.py --dataset /workspace/data/demo1 \
                                           --checkpoint /workspace/outputs/checkpoints/demo1_xxx \
                                           --policy diffusion
```

## 3. Run inference scripts

Once inside the container, you can execute the inference node:

```bash
ros2 run franka_ai_inference inference_node
```

Every time a new modification is done in the mounted inference_node.py, you need to run:

```bash
colcon build
source install/setup.bash
```

Otherwise, you can just run the first time inside Docker the following:

```bash
colcon build --symlink-install --packages-select franka_ai_inference
```

## 4. Known issues

## 5. References

- **A. Xie, O. Rybkin (2025)**: [*Latent Diffusion Planning for Imitation Learning*](https://arxiv.org/pdf/2504.16925)
- **E. Nava, V. Montesinos (2025)**: [*Mimic-One: a Scalable Model Recipe for General Purpose Robot Dexterity*](https://arxiv.org/pdf/2506.11916)
- **Tony Z. Zhao (2023)**: [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/pdf/2304.13705)
- **R. Cadene, Alibert (2023)**: [*LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch*](https://github.com/huggingface/lerobot)

## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/leo-berte/franka_AI/blob/main/LICENSE) file for details.
