# Run Deepseek R1 2.51 bit on 8 * W7800 48G System

# 1. Download the model

For 8 * 48G System, we use the [unsloth 2.51 bit model](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL), which takes about 230G GPU memory, leaving sufficient GPU memory for kvcahe.

We recommand using the hfd tool provided by hf-mirror to download the model:

```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
./hfd.sh unsloth/DeepSeek-R1-GGUF --include DeepSeek-R1-UD-Q2_K_XL
```

If you are in China, add this step before running the above command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

# 2. Convert the model to a single file

vLLM only supports single file GGUF model. We shoud use `llama-gguf-split` tool from llama.cpp to convert the model:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
export LLAMA_CPP_HOME=$(pwd)
cmake -B build && cmake --build build -j
$LLAMA_CPP_HOME/build/bin/llama-gguf-split \
    --merge <path_to_your_model> \
    <path_to_output_dir>/DeepSeek-R1-UD-Q2_K_XL.gguf
```

# 3. Install dependencies

## Crate new conda environment

```bash
mamba create -n vllm-rocm python=3.12
```

## Install pytorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Or you can use pytorch image from [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch) to run the following commands.

## Install transformers

Transformers must be patched to run Deepseek R1 GGUF model. `0001-feat-add-deepseek-v2-support-for-gguf-models.patch` can be found in this folder.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
cp ../0001-feat-add-deepseek-v2-support-for-gguf-models.patch .
git apply 0001-feat-add-deepseek-v2-support-for-gguf-models.patch
pip3 install -e .
```

## Install other dependencies

flash-attention can be installed from source. Note, you should set `GPU_ARCHS` to your GPU architecture, which can be found in [AMD ROCm documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). For W7800, W7900 or 7900XTX, it is `gfx1100`.

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" GPU_ARCHS="gfx1100" pip install -e .
```

## Install vLLM

```bash
export ROCM_VLLM_HOME=$(pwd)/vllm
git clone https://github.com/vllm-project/vllm.git
cd /opt/rocm/share/amd_smi
pip install .
cd ${ROCM_VLLM_HOME}
pip install --upgrade numba \
    scipy \
    huggingface-hub[cli,hf_transfer] \
    setuptools_scm
pip install "numpy<2"
pip install -r requirements/rocm.txt
export PYTORCH_ROCM_ARCH="gfx1100"
python use_existing_torch.py
python3 setup.py develop
```

# 4. Run vLLM

You can run vLLM with the following command:

```bash
 FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" vllm serve /hpc/syh/DeepSeek-R1-UD-Q2_K_XL.gguf --tokenizer deepseek-ai/DeepSeek-R1 --enable-prompt-tokens-details --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel 8 --gpu-memory-utilization 0.95
```
