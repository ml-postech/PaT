# PaT: Planning-after-Trial for Efficient Test-Time Code Generation

This repository contains the implementation of the **Planning-after-Trial (PaT)** framework, designed for robust and efficient code generation.

---

### **Acknowledgment**

The majority of the code in this repository is based on the [**FunCoder**](https://github.com/cometeme/funcoder) project. We extend our deepest gratitude to the original authors for their foundational work, which served as a crucial basis for our research.

---

### **Setup**

To run experiments, you need to set up

- Environment

conda create -y -n PaT python 3.10
conda activate PaT
python -m pip install -r requirements.txt


- Datasets

python -m PaT.eval download-datasets

- Configuration

python -m vllm.entrypoints.openai.api_server --model /path/to/your/model/Qwen3-8B --dtype float16 --api-key token-qwen3_8 --port 28110

### **Experiments**

python -m Pat.eval draft --results-dir /your/experiment/dir/ 

python -m Pat.eval judge --results-dir /your/experiment/dir/
