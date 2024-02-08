# ROCm tutorials

A collection of tutorials for using ROCm.

## Collections

### Websties

- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/): Main documentation for ROCm, all about its components and how to use them.
- [GPUOpen](https://gpuopen.com): A collection of resources from AMD and GPUOpen partners, including ISA documentation, developer tools, libraries, and SDKs.

ROCm have a lot of github Organizations and Repositories, here are some of them:

- [ROCm Core Technology](https://github.com/RadeonOpenCompute): Low level drivers and runtimes for ROCm.
- [ROCm Developer Tools](https://github.com/ROCm-Developer-Tools): Tools for profiling, debugging, and optimizing applications for ROCm.
- [ROCm Software Platform](https://github.com/ROCmSoftwarePlatform): High level libraries and frameworks for ROCm, like Pytorch, Tensorflow, MIOpen, etc. Xformers and Flash-attention are also here.

The docker hub for ROCm is [rocm](https://hub.docker.com/u/rocm), you can find all the official docker images here.

### Useful Repositories

- [HIPIFY](https://github.com/ROCm/HIPIFY): A tool to convert CUDA code to HIP code. You can use it to port your CUDA code to ROCm.
- [Flash-Attention](https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm)

#### Inference
- [FastLLM-ROCm](https://github.com/lcpu-club/fastllm-rocm/tree/master): A simple implementation of FastLLM on ROCm. Not optimized, but it is easy to maintain and modify.
- [VLLM](https://github.com/vllm-project/vllm): A high performance implementation of FastLLM on ROCm. It is optimized for performance. It have [AMD Installation Guide](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html) and [Docker image](https://hub.docker.com/r/embeddedllminfo/vllm-rocm/tags) for MI GPUs.

#### Training and Fine-tuning
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning from Huggingface, very easy to use
- [Deepspeed](https://github.com/microsoft/DeepSpeed): Distributed training and inference library.

#### Quantization

- [BitsAndBytes](https://github.com/lcpu-club/bitsandbytes-rocm): 8-bit CUDA functions for PyTorch, ported to HIP for use in AMD GPUs. 4 bit is on the way.

## Installation and environment setup

See env-install folder for useful scripts to install ROCm and setup environment. All of the scripts need Pytorch to run, so you need to install Pytorch first.

- test-rocm.py: A script to test if ROCm is installed correctly.
- test-pytorch.py: A script to test performance of Pytorch on ROCm using GEMM operations.

### Steps to install ROCm

1. Check system compatibility: [ROCm System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
2. Install via package manager: [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html)

Note, after installing the AMDGPU driver, a reboot is required.

### Steps to install Pytorch

#### From Docker image

The easiest way to install Pytorch is to use the docker image provided by ROCm. You can find the docker hub [here](https://hub.docker.com/u/rocm). We use `rocm/pytorch:rocm6.0_ubuntu22.04_py3.9_pytorch_2.0.1`, you can pull it by:

```bash
docker pull rocm/pytorch:rocm6.0_ubuntu22.04_py3.9_pytorch_2.0.1
```

Check out the [Docker tutorial for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) for more information about GPUs control and docker images.

If you just want to run Pytorch on ROCm, the command is:

```bash
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G rocm/pytorch:rocm6.0_ubuntu22.04_py3.9_pytorch_2.0.1
```

#### From wheel

If you want to install Pytorch on your host machine, you can install it from wheel. You can find the steps on Pytorch official website: [Pytorch Get Started](https://pytorch.org/get-started/locally/).

An example command to install Pytorch on ROCm is:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
```

Note: virtual environment is recommended. We recommand mamba to create virtual environment, it is much faster than conda.

## Fine-tuning example

See fine-tuning folder for fine-tuning examples. You can run it by `bash run.sh`.

Some notes:

1. BitAndBytes is used to quantize the model to 8-bit, but its ported haven't finished yet. You need to use `adamw_torch` optimizer to avoid error.
2. The base model is `meta-llama/Llama-2-7b-chat-hf`, you shoud get access to it first.
