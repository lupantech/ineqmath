# **IneqMath**: A Benchmark for Informal, Verifiable Reasoning in Olympiad-Level Inequality Proofs

Code for the Paper [Solving Inequality Proofs with Large Language Models (TODO)](https://www.google.com)

<p>
    <a href="https://ineqmath.github.io/">🌐 Project</a> |
    <a href="https://www.google.com">📖 Paper (TODO)</a> |
    <a href="https://huggingface.co/datasets/AI4Math/IneqMath">🤗 Dataset</a> |
    <a href="https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard">🏆 Leaderboard</a>
  </p>

## 🏆 Leaderboard
The leaderboard of chat and reasoning LLMs on the **IneqMath** benchmark (the test set) is shown below. 

The interactive leaderboard for the **IneqMath** is available [here](https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard).

| **Rank** | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Model**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;**Size**&nbsp;&nbsp;&nbsp; | **Type** | **Source** | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Date**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **Overall Acc ↓** | **Answer Acc** | **Step Acc (NTC)** | **Step Acc (NLG)** | **Step Acc (NAE)** | **Step Acc (NCE)** |
|------|-------|------|------|--------|------|-------------|------------|----------------|----------------|----------------|----------------|
| 1 | **Gemini 2.5 Pro (30K)🥇** | UNK | 🧠 | 🔒 | 2025-03-25 | **43.5** | 68.0 | 87.5 | 63.0 | 91.0 | 98.0 |
| 2 | **o3 (medium, 40K)🥈** | UNK | 🧠 | 🔒 | 2025-04-16 | **37.0** | 72.0 | 96.5 | 56.0 | 86.5 | 94.0 |
| 3 | **Gemini 2.5 Flash (40K)🥉** | UNK | 🧠 | 🔒 | 2025-04-17 | **23.5** | 44.5 | 81.0 | 36.5 | 93.5 | 97.5 |
| 4 | **o3 (medium)** | UNK | 🧠 | 🔒 | 2025-04-16 | **21.0** | 37.0 | 93.5 | 39.5 | 91.5 | 97.0 |
| 5 | **o4-mini (medium)** | UNK | 🧠 | 🔒 | 2025-04-16 | **15.5** | 65.0 | 62.0 | 26.0 | 86.5 | 93.0 |
| 6 | **o3-mini (medium)** | UNK | 🧠 | 🔒 | 2025-01-31 | **9.5** | 62.5 | 37.0 | 22.0 | 77.5 | 95.0 |
| 7 | **o1 (medium)** | UNK | 🧠 | 🔒 | 2024-12-17 | **8.0** | 62.5 | 34.5 | 17.5 | 86.5 | 99.5 |
| 8 | **o1 (medium, 40K)** | UNK | 🧠 | 🔒 | 2024-12-17 | **7.5** | 68.0 | 28.5 | 19.0 | 83.5 | 95.5 |
| 9 | **Grok 3 mini (medium)** | UNK | 🧠 | 🔒 | 2025-02-19 | **6.0** | 71.5 | 24.0 | 19.5 | 53.5 | 91.0 |
| 10 | **Qwen3-235B-A22B** | 235B | 🧠 | 🌐 | 2025-04-28 | **6.0** | 41.0 | 35.0 | 36.0 | 31.0 | 92.5 |
| 11 | **Gemini 2.5 Pro** | UNK | 🧠 | 🔒 | 2025-03-25 | **6.0** | 7.0 | 88.5 | 19.0 | 100.0 | 99.5 |
| 12 | **DeepSeek-R1 (Qwen-14B)** | 14B | 🧠 | 🌐 | 2025-01-20 | **5.0** | 40.5 | 21.0 | 21.0 | 35.5 | 85.0 |
| 13 | **DeepSeek-R1** | UNK | 🧠 | 🔒 | 2025-01-19 | **5.0** | 49.5 | 57.0 | 17.5 | 81.0 | 95.0 |
| 14 | **Gemini 2.5 Flash** | UNK | 🧠 | 🔒 | 2025-04-17 | **4.5** | 5.5 | 88.0 | 13.5 | 100.0 | 100.0 |
| 15 | **Grok 3** | UNK | 📝 | 🔒 | 2025-02-19 | **3.5** | 54.5 | 17.0 | 16.0 | 36.0 | 93.0 |
| 16 | **DeepSeek-R1 (Llama-70B)** | 70B | 🧠 | 🌐 | 2025-01-20 | **3.5** | 53.5 | 23.0 | 26.0 | 35.5 | 87.0 |
| 17 | **Gemini 2.0 Flash** | UNK | 📝 | 🔒 | 2025-02-05 | **3.0** | 49.0 | 15.5 | 13.5 | 55.5 | 94.5 |
| 18 | **Qwen2.5-7B** | 7B | 📝 | 🌐 | 2024-09-16 | **3.0** | 35.0 | 44.5 | 4.5 | 92.5 | 93.0 |
| 19 | **GPT-4o** | UNK | 📝 | 🔒 | 2024-08-06 | **3.0** | 37.5 | 32.0 | 3.5 | 92.5 | 94.0 |
| 20 | **GPT-4.1** | UNK | 📝 | 🔒 | 2025-04-14 | **2.5** | 40.5 | 16.0 | 10.0 | 59.5 | 93.5 |
| 21 | **Qwen2.5-72B** | 72B | 📝 | 🌐 | 2024-09-16 | **2.5** | 42.0 | 54.5 | 5.0 | 91.0 | 95.0 |
| 22 | **Llama-4-Maverick** | 128 x 17B | 📝 | 🌐 | 2025-04-05 | **2.5** | 40.5 | 42.5 | 4.0 | 89.0 | 95.0 |
| 23 | **Claude 3.7 Sonnet** | UNK | 🧠 | 🔒 | 2025-02-19 | **2.0** | 42.0 | 49.0 | 4.0 | 93.5 | 93.0 |
| 24 | **QwQ-32B** | 32B | 🧠 | 🌐 | 2025-03-05 | **2.0** | 49.5 | 26.0 | 29.5 | 21.0 | 87.0 |
| 25 | **QwQ-32B-preview** | 32B | 🧠 | 🌐 | 2024-11-27 | **2.0** | 43.5 | 28.0 | 30.0 | 22.5 | 87.5 |
| 26 | **GPT-4o mini** | UNK | 📝 | 🔒 | 2024-07-18 | **2.0** | 39.5 | 29.0 | 2.5 | 90.0 | 93.0 |
| 27 | **Qwen2.5-Coder-32B** | 32B | 📝 | 🌐 | 2024-11-10 | **1.5** | 40.5 | 36.0 | 3.0 | 90.5 | 88.5 |
| 28 | **Gemini 2.0 Flash-Lite** | UNK | 📝 | 🔒 | 2025-02-25 | **1.5** | 33.0 | 11.5 | 3.5 | 73.0 | 90.5 |
| 29 | **Qwen2.5-Coder-32B** | 32B | 📝 | 🌐 | 2024-11-10 | **1.5** | 40.5 | 36.0 | 3.0 | 90.5 | 88.5 |
| 30 | **Llama-4-Scout** | 16 x 17B | 📝 | 🌐 | 2025-04-05 | **1.5** | 33.5 | 30.5 | 3.5 | 93.0 | 92.5 |
| 31 | **Claude 3.7 Sonnet (8K)** | UNK | 🧠 | 🔒 | 2025-02-19 | **1.0** | 41.5 | 49.0 | 2.5 | 93.5 | 92.0 |
| 32 | **DeepSeek-R1 (Qwen-14B)** | 1.5B | 🧠 | 🌐 | 2025-01-20 | **0.5** | 14.5 | 20.0 | 6.0 | 48.0 | 83.5 |
| 33 | **Gemma-2B (6K)** | 2B | 📝 | 🌐 | 2024-02-21 | **0.0** | 7.5 | 73.5 | 0.0 | 99.0 | 95.0 |
| 34 | **Llama-3.1-8B** | 8B | 📝 | 🌐 | 2024-07-18 | **0.0** | 14.5 | 90.5 | 0.0 | 99.0 | 92.0 |
| 35 | **Gemma-2-9B (6K)** | 9B | 📝 | 🌐 | 2024-06-25 | **0.0** | 15.5 | 83.5 | 0.5 | 100.0 | 99.0 |
| 36 | **Llama-3.2-3B** | 3B | 📝 | 🌐 | 2024-09-25 | **0.0** | 11.0 | 82.0 | 0.0 | 98.5 | 88.5 |

Icons Explanation:
**Type:** 🧠 = Reasoning Model, 📝 = Chat Model, 🔧 = Tool-augmented Model
Source: 🔒 = Proprietary Model, 🌐 = Open-source Model

**Step Accuracy Abbreviations:**
**NTC**: No Toy Case - Step accuracy excluding using toy-case for general conclusions
**NLG**: No Logical Gap - Step accuracy without logical reasoning gaps
**NAE**: No Approximation Error - Step accuracy excluding approximation errors
**NCE**: No Calculation Error - Step accuracy excluding all calculation errors



## About IneqMath
IneqMath is a benchmark for evaluating large language models (LLMs) on informal but verifiable inequality proving. Centered on Olympiad-level algebraic inequalities, it challenges models to not only produce correct final answers but also construct step-by-step solutions that apply theorems appropriately, justify symbolic transformations, and estimate tight bounds. Problems are framed in natural language and decomposed into two automatically checkable subtasks—bound estimation and relation prediction—allowing fine-grained assessment of reasoning accuracy beyond surface-level correctness.

### Dataset Overview
The table below provides the statistics of **IneqMath**, along with the bound and relation subtasks.
<center>
<small>
  <table 
    align="center" 
    width="60%" 
    border="1" 
    cellspacing="0" 
    cellpadding="6"
    style="width:60%; table-layout: fixed; border-collapse: collapse; text-align: center; font-size: 0.6em;">
    <colgroup>
      <col width="64%">
      <col width="12%">
      <col width="12%">
      <col width="12%">
    </colgroup>
    <thead>
      <tr>
        <th style="text-align:left;">Statistic</th>
        <th>Number</th>
        <th>Bnd.</th>
        <th>Rel.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align:left;"><b>Theorem categories</b></td>
        <td>29</td>
        <td>–</td>
        <td>–</td>
      </tr>
      <tr style="border-bottom:2px solid #000;">
        <td style="text-align:left;"><b>Named theorems</b></td>
        <td>83</td>
        <td>–</td>
        <td>–</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Training problems (for training)</b></td>
        <td>1252</td>
        <td>626</td>
        <td>626</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- With theorem annotations</td>
        <td>962</td>
        <td>482</td>
        <td>480</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- With solution annotations</td>
        <td>1252</td>
        <td>626</td>
        <td>626</td>
      </tr>
      <tr>
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Avg. solutions per problem</td>
        <td>1.05</td>
        <td>1.06</td>
        <td>1.05</td>
      </tr>
      <tr style="border-bottom:2px solid #000;">
        <td style="text-align:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Max solutions per problem</td>
        <td>4</td>
        <td>4</td>
        <td>4</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Dev problems (for development)</b></td>
        <td>100</td>
        <td>50</td>
        <td>50</td>
      </tr>
      <tr>
        <td style="text-align:left;"><b>Test problems (for benchmarking)</b></td>
        <td>200</td>
        <td>96</td>
        <td>104</td>
      </tr>
    </tbody>
  </table>
  </small>
</center>

<br><br>

The chart below shows the distribution of theorem categories.

<br><br>

<div align="center">

  <img src="./assets/theorem_category_pie_chart.png" alt="IneqMath Logo" width="650"/>

</div>

## Environment Setup

Set up conda environment:

```bash
conda create --name ineq python=3.10

conda activate ineq

```
If you fail to activate the environment, please try:
```bash
# Alternatively, use source activate
source activate ineq
```



Install dependencies and make `.env` file:

```bash
pip install -r requirements.txt
touch .env
```


Set your API keys in the `.env` file. For example:

```sh
OPENAI_API_KEY=your-OpenAI-api-key-here
DEEPSEEk_API_KEY=your-DeepSeek-api-key-here
ANTHROPIC_API_KEY=your-Anthropic-api-key-here
```

Finally, please remove all .DS_Store files:
```bash
find . -name ".DS_Store" -delete
```

## Evaluate models on **IneqMath** test set
Change the directory to `models/scripts`:
```bash
cd models/scripts
```

Run `run_test_data_proprietary_all.sh`, `run_test_data_open_source_all.sh`, and `run_test_data_gemma.sh` to evaluate all the models used in our paper's experiments on the test set.
```bash
./run_test_data_proprietary_all.sh
./run_test_data_open_source_all.sh
./run_test_data_gemma.sh
```

If the dataset can't be loaded automatically, please download the json form dataset manually by:
```shell
mkdir ../../data
cd ../../data
wget https://huggingface.co/datasets/AI4Math/IneqMath/resolve/main/json/all.tar.gz
tar -zxvf all.tar.gz
```

If you want to run other models on our test set, you could subtitute the model engine name in `ENGINES` of the `.sh` file, and then run it.






## Submit the results to the leaderboard
🏆 The leaderboard for the **IneqMath** is available [here](https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard).

If you run the model by our scripts, you can find the results in `results models_results_test_data/` and upload the `results.json` of the model to the leaderboard.

If you run the model on your own, please check your data format before your submission. The submitted data should be compiled in a single `json` file with at least five keys listed below:

```
{
    "data_id": [integer] The ID of the data of each split,
    "problem": [string] The question text,
    "type": [string] The type of question: 'relation' or 'bound',
    "prompt": [string] The prompt used for the problem,
    "response": [string] The response of the model
}
```
# Dataset Examples
Training examples of **IneqMath**:
<div align="center">
    <img src="assets/train_bound_example.png" width="650" alt="Train Bound Example">
    <img src="assets/train_relation_example.png" width="650" alt="Train Relation Example">
</div>

<br><br>

Testing examples of **IneqMath**:

<br><br>

<div align="center">
    <img src="assets/test_bound_example.png" width="650" alt="Test Bound Example">
    <img src="assets/test_relation_example.png" width="650" alt="Test Relation Example">
</div>

# LLM Judge Performance

Confusion matrices and performance metrics of our 5 LLM-as-Judges are shown below, which exhibit strong agreement with human labels.

<div align="center">
  <img src="./assets/confusion_matrix_judge_metrix.png" alt="judge_confusion_matrix" width="800"/>
  <img src="./assets/table_judge_metrics.png" alt="table_judge_matrix" width="650"/>

</div>

# Scaling law in model size
The following two figures show how <em>final-answer accuracy</em> (which evaluates only the correctness of the final predicted answer) and <em>overall accuracy</em> (which requires both a correct answer and valid intermediate reasoning steps) scales with model size for LLMs.

<div align="center">

  <img src="./assets/scaling_law_model_size_answer_acc_log_all.png" alt="scaling_curve_answer_acc" width="650"/>
  <img src="./assets/scaling_law_model_size_overall_acc_log_all.png" alt="scaling_curve_overall_acc" width="650"/>

</div>

# License

The new contributions to our dataset are distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The copyright of the images and the questions belongs to the original authors. Alongside this license, the following conditions apply:

- **Purpose:** The test split was primarily designed for use as a test set.
- **Commercial Use:** The test split can be used commercially as a test set, but using it as a training set is prohibited. By accessing or using this dataset, you acknowledge and agree to abide by these terms in conjunction with the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

# Citation

If you use the **IneqMath** dataset in your work, please kindly cite the paper using this BibTeX:

```
TODO
```

