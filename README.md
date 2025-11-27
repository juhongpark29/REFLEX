   # REFLEX Accelerator Simulator
   
   This repository contains the **REFLEX Simulator**, used to evaluate efficient long-context sparse attention under PIM-based execution.
   
   The simulator consists of two main components:
   
   1. **Page Selection Frontend**  
      Uses the *OmniServe (LServe)*—a state-of-the-art mixed sparse attention method—to generate score-guided KV page selections.  
      REFLEX enables testing different combinations of **WMₙ (write modulus)** and **RMₘ (read modulus)** to evaluate their impact on long-context accuracy.
   
   2. **PIM Execution Backend**  
      Extends the **AttAcc** simulator on **Ramulator 2.0 (HBM3)** with REFLEX-specific row-aligned scheduling.  
      This backend models PIM execution, capturing activation behavior and memory-cycle performance.
   
   > **Note**  
   > This repository is provided *exclusively for reviewers* to reproduce the evaluation setup for our submission.  
   > Additional implementation details will be released after publication.
   
   ---
   # How to Install
   
   REFLEX depends on the following external components:
   
   1. **OmniServe (LServe)** – score-guided sparse page selection  
   2. **Block-Sparse-Attention** – sparse KV prefilling  
   3. **AttAcc (Ramulator2)** – PIM execution simulation  
   
   REFLEX **extends** these components with additional modules for row-aligned page selection and PIM-side execution. External repositories are **not bundled** and must be installed separately.
   
   
   ## Reference Links
   
   - **OmniServe**  
     https://github.com/mit-han-lab/OmniServe
   
   - **Block-Sparse-Attention**  
     https://github.com/mit-han-lab/Block-Sparse-Attention
   
   - **AttAcc (Ramulator2)**  
     https://github.com/scale-snu/attacc_simulator
   
   ---
   
   ## 1. Install OmniServe (LServe-based Sparse Attention)
   
   ```bash
   git clone https://github.com/mit-han-lab/OmniServe
   cd OmniServe
   
   conda create -n OmniServe python=3.10 -y
   conda activate OmniServe
   pip install --upgrade pip
   ```
   ```bash
   # (optional) install CUDA toolkit if nvcc is not available
   conda install -c nvidia cuda-toolkit -y
   ```
   ```bash
   # Install OmniServe package
   pip install -e .
   pip install flash-attn --no-build-isolation
   ```
   
   To verify installation, start a Python session and run:
   
   ```python
   import flash_attn
   ```
   
   ## 2. Install Block-Sparse-Attention (for Sparse Prefilling)
   
   Block-Sparse-Attention is required for sparse KV prefilling.
   
   ```bash
   git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git --recursive
   cd Block-Sparse-Attention
   
   pip install packaging ninja
   python setup.py install
   ```
   
   Compile kernels inside OmniServe:
   ```bash
   cd ../OmniServe/kernels
   python setup.py install
   ```
   
   ## 3. Install AttAcc (PIM Execution Backend)
   ```bash
   git clone https://github.com/scale-snu/attacc_simulator.git
   cd attacc_simulator
   git submodule update --init --recursive
   ```
   
   Build Ramulator2 (modified for AttAcc):
   ```bash
   bash set_pim_ramulator.sh
   cd ramulator2
   mkdir build && cd build
   cmake ..
   make -j
   cp ramulator2 ../ramulator2
   cd ../../
   ```
   
   ## 4. Install REFLEX Simulator
   
   Clone this anonymous repository, then run the patch script:

   ```bash
   git clone https://anonymous.4open.science/r/REFLEX-55B9
   cd /workspace/REFLEX

   bash setup_workspace.sh /workspace
   ```
   This script copies the REFLEX patch files into the existing workspace and overwrites the corresponding files in `OmniServe/` and `attacc_simulator/`.
   
   After running the script, your directory layout will look like:
   ```bash
   /workspace
   ├─ OmniServe/
   ├─ Block-Sparse-Attention/
   ├─ attacc_simulator/
   └─ REFLEX/
      ├─ OmniServe/
      ├─ AttAcc/
      └─ setup_workspace.sh/
   ```

   # How to run

   ## 1. Sparse KV Selection
   Enable logging to export selected page indices:
   
   ```bash
   export ENABLE_PAGE_SELECTION_LOG=true
   ```
   Run the NIAH-based end-to-end LServe generation:
   ```bash
   cd /workspace/OmniServe
   bash scripts/lserve_e2e.sh
   ```
   This generates CSV files containing the selected page indices under the RMₘ page-selection rules described in our paper:
   ```bash
   $PAGE_SELECTION_LOG_DIR   (default: ./my_page_logs)
   ```
   These CSV files are used as input for PIM simulation.

   > **Note**
   > To reproduce the accuracy results reported in the paper—or to explore different page-selection settings (e.g., WMₙ/RMₘ configurations or sparsity levels)—you can use the following evaluation scripts:
   > ```bash
   > cd /workplace/OmniServe
   > bash eval/scripts/needle/submit_niah.sh
   > bash eval/scripts/LongBench/submit_longbench.sh
   > ```
   > These run the accuracy evaluation for NIAH and LongBench under the same page-selection setup used in our paper.

   ## 2. PIM simulation 
   REFLEX supports two workflows:

   ### 2.1 PIM-only configuration (GEMV)
   This pipeline converts the LServe CSV log into a PIM trace, and runs the simulation on Ramulator2.

   > **Note**  
   > LServe logs reflect only the RMₘ rule. To apply the WMₙ rule as well, convert the page indices using the `page_converter.py` script, which maps RMₘ-selected pages to their WMₙ counterparts:  
   > ```bash
   > /workspace/attacc_simulator/ramulator2/trace_gen/page_converter.py
   > ```  

   
   (1) Convert LServe CSV into an PIM trace
   
   The CSV used here is the WMₙ-converted version of the LServe log (originally RMₘ-only) and reflects the WM8RM8 configuration used to generate the PIM trace.
   The PIM trace is produced using the `gen_trace_ori.py` script:
   
   ```python
   cd /workspace/attacc_simulator/ramulator2/trace_gen && \
   python gen_trace_ori.py \
    --csv mod8_4_page_selection_log_20251013_193934_selected_pages.csv \
    --output ori_buffer_mod8_timestep_first.trace \
    --seqlen 21504 \
    --nhead 32 \
    --page_size 64
   ```
   This produces:
   - ori_buffer_mod8_timestep_first.trace
   
   (2) Run Ramulator2 with the generated trace
   ```python
   cd /workspace/attacc_simulator/ramulator2
   ./ramulator2 -f ori_buffer_mod8.yaml
   ```
   (3) Analyze the simulation output
   
   Use the `analyze_trace.py` script to compute latency and activation statistics from the generated trace:
   
   ```python
   cd /workspace/attacc_simulator/
   python analyze_trace.py \
       --trace_file ramulator2/trace_gen/ori_buffer_mod8_timestep_first.trace \
       --seqlen 21504 \
       --output ori_mod8_nopipe_report
   ```

   ### 2.2 Hybrid configuration: GPU (GEMM) + PIM (GEMV)
   This pipeline automatically performs the entire workflow—CSV parsing, trace generation, and PIM simulation—through `main.py`. You only need to provide the **WMₙ-converted CSV**.

   > **Note**  
   > Setting `--page_scheduling true` enables the REFLEX row-aligned command scheduling used in our paper.

   ```bash
   cd /workspace/attacc_simulator && \
   python main.py \
    --system dgx-attacc \
    --gpu A100a \
    --ngpu 8 \
    --model LLAMA-8B \
    --lin 21504 \
    --batch 64 \
    --pim buffer \
    --csv_file "mod8_4_page_selection_log_20251013_193934_selected_pages.csv" \
    --output_trace "ori_buffer_mod8_timestep_first.trace" \
    --page_scheduling false \
    --mac_mode bank \
    --page_mode ori
   ```
   
