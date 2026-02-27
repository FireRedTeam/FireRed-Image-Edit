<p align="center">
    <img src="./assets/logo.png" width="600"/>
<p> 
<p align="center">
  <a href="https://huggingface.co/FireRedTeam" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRedTeam-ffc107?color=ffc107&logoColor=white" style="display: inline-block;"/></a>
  <a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRed--Image--Edit--1.0-red" style="display: inline-block;"/></a>
  <a href="https://huggingface.co/spaces/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="HF Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Demo-FireRed--Image--Edit--1.0-red" style="display: inline-block;"/></a>
  <a href="https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="ModelScope Model" src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-FireRed--Image--Edit--1.0-624aff" style="display: inline-block;"/></a>
  <a href="https://modelscope.cn/studios/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="MS Demo" src="https://img.shields.io/badge/%F0%9F%A4%96%20MS%20Demo-FireRed--Image--Edit--1.0-624aff" style="display: inline-block;"/></a>
  <a href='https://github.com/FireRedTeam/FireRed-Image-Edit'><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
  <a href='https://www.apache.org/licenses/LICENSE-2.0'><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://arxiv.org/abs/2602.13344" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv"></a>
</p> 

<p align="center">
    🤗 <a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0">HuggingFace</a> |
    🤖 <a href="https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0">ModelScope</a> |
    🖥️ <a href="https://huggingface.co/spaces/FireRedTeam/FireRed-Image-Edit-1.0">Demo</a> |
    📄 <a href="https://arxiv.org/abs/2602.13344">Technical Report</a>
</p>
<p align="center">
    <img src="./assets/teaser.png" width="800"/>
<p> 


## 🔥 FireRed-Image-Edit
**FireRed-Image-Edit** is a general-purpose image editing model that delivers high-fidelity and consistent editing across a wide range of scenarios.

## ✨ Key Features
- **Strong Editing Performance**: FireRed-Image-Edit delivers leading open-source results with accurate instruction following, high image quality, and consistent visual coherence.
- **Native Editing Capability from T2I Backbone**: Built upon an open-source text-to-image foundation model , we introduce editing ability through a full pipeline of **Pretrain**, **SFT**, and **RL**. This training paradigm is backbone-agnostic and can be applied to other T2I models(currently based on Qwen-Image for better community support). We will progressively open-source additional features, including self-developed FireRed T2I foundation model later.
- **Text Style Preservation**: Maintains text styles with high fidelity, achieving performance comparable to closed-source solutions.
- **Photo Restoration**: High-quality old photo restoration and enhancement.
- **Multi-Image Editing**: Flexible editing of multiple images such as virtual try-on.


## 📰 News
- 2026.02.27: We released the [Agent](#-agent) module for instruction rewriting, multi-image preprocessing, supporting automatic ROI detection, image stitching for editing with more than 3 images.
- 2026.02.14: We released FireRed-Image-Edit-1.0 model weights. Check more details on [Huggingface](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0) and [ModelScope](https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0).
- 2026.02.10: We released the [Technical Report](https://arxiv.org/abs/2602.13344) of FireRed-Image-Edit-1.0. 

## 🏆 Evaluation Results
FireRed-Image-Edit establishes a new state-of-the-art among open-source models on Imgedit, Gedit, and RedEdit, while surpassing our closed-source competitors in specific dimensions—a distinction further corroborated by human evaluations highlighting its superior prompt following and visual consistency.
<p align="center">
    <img src="./assets/eval_benchmark.png" width="600"/>
<p>
<p align="center">
    <img src="./assets/eval_human.png" width="600"/>
<p>


## 🎨 Showcase
Some real outputs produced by FireRed-Image-Edit across general editing.
<p align="center">
    <img src="./assets/showcase.png" width="800"/>
<p> 

* Creative scenarios
<p align="center">
    <img src="./assets/showcase_creativity.png" width="800"/>
<p> 

* Text scenarios:
<p align="center">
    <img src="./assets/showcase_text.png" width="800"/>
<p> 

* Tryon scenarios:
<p align="center">
    <img src="./assets/showcase_virtual_tryon.png" width="800"/>
<p> 

## 🗂️ Model Zoo

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Models</th>
        <th style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Task</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Description</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa;">Download Link</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image-Edit-1.0</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image-Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">General-purpose image editing model</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0">HuggingFace</a></span>
          <span style="white-space: nowrap;">🤖&nbsp;<a href="https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0">ModelScope</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image-Edit-1.0-Distilled</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image-Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Distilled version of FireRed-Image-Edit-1.0 for faster inference</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">To be released</span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text-to-Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">High-quality text-to-image generation model</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">To be released</span>
        </td>
      </tr>
    </tbody>
  </table>
</div>

## 🏗️ Model Architecture
<p align="center">
    <img src="./assets/architecture.png" width="800"/>
<p> 

## ⚡️ Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Use the following code snippets to generate or edit images.
```
python inference.py \
    --input_image ./examples/edit_example.png \
    --prompt "在书本封面Python的下方，添加一行英文文字2nd Edition" \
    --output_image output_edit.png \
    --seed 43
```

## 🤖 Agent

The Agent module provides **Recaption & Multi-Image Preprocessing** capabilities.

FireRed-Image-Edit natively supports **1–3** input images. When users need to edit with **more than 3 images**, the built-in **Agent** module automatically:

1. **ROI Detection** – Sends all images + the user instruction to a Gemini function-calling model that returns a bounding-box for the most relevant region in each image.
2. **Crop & Stitch** – Crops each image to its ROI, then partitions and stitches them into **2–3 composite images** (≈1024×1024 each) while minimising whitespace and preserving content at maximum resolution.
3. **Recaption** – Rewrites the user instruction so that image references (图1/图2/image N …) correctly point to the new composite images, and expands the prompt to ~512 words/characters for richer editing context. The user's original language is preserved.


**(Optional)** To enable the **Recaption** feature (rewriting instructions via Gemini for better editing results), set your Gemini API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

> **Note:** The Gemini API is **not required**. Without it, the Agent will still perform ROI detection and image stitching normally, but will skip the instruction rewriting step. Setting a Gemini API key is recommended for best results.


### Agent File Structure

```
agent/
├── __init__.py        # Package entry – exports AgentPipeline
├── config.py          # Configuration (API keys, stitch parameters, etc.)
├── gemini_agent.py    # Gemini function-calling for ROI detection
├── image_tools.py     # Image tools: crop, resize, stitch, partition
├── recaption.py       # Instruction rewriting & expansion via Gemini
└── pipeline.py        # End-to-end orchestration pipeline
```

## 📊 Benchmark
To better validate the capabilities of our model, we propose a benchmark called REDEdit-Bench. Our main goal is to build more diverse scenarios and editing instructions that better align with human language, enabling a more comprehensive evaluation of current editing models. We collected over 3,000 images from the internet, and after careful expert-designed selection, we constructed 1,673 bilingual (Chinese–English) editing pairs across 15 categories.

### Inference and Evaluation Code
We provide the inference and evaluation code for REDEdit-Bench. Please refer to the [redbench_infer.py](./tools/redbench_infer.py) and [redbench_eval.py](./tools/redbench_eval.py) scripts in the `tools` directory for more details.

### Benchmark Distribution
The REDEdit-Bench dataset will be available soon.

### Results on ImgEdit

<table style="border-collapse:collapse; width:100%; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; font-size:14px;">
  <thead>
    <tr>
      <th style="border:1px solid #ddd; padding:8px; text-align:left;">Model</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Overall ↑</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Add</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Adjust</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Extract</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Replace</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Remove</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">BG</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Style</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Hybrid</th>
      <th style="border:1px solid #ddd; padding:8px; text-align:center;">Action</th>
    </tr>
  </thead>
  <tbody><tr>
      <td colspan="11" style="border:1px solid #ddd; padding:8px; font-weight:700; background:#f7f7f7;">
        🔹 Proprietary Models
      </td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Nano-Banana</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.29</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.62</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.41</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>3.68</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.34</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.39</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.40</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.18</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>3.72</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.83</b></td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Seedream4.0</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.30</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.33</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.38</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>3.89</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.65</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.57</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.35</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.22</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>3.71</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.61</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Seedream4.5</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.32</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.57</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.65</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">2.97</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.66</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.46</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.37</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.92</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>3.71</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.56</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Nano-Banana-Pro</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.37</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.44</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.62</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.42</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.60</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.63</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.32</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.97</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.64</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.69</u></td>
    </tr><tr>
      <td colspan="11" style="border:1px solid #ddd; padding:8px; font-weight:700; background:#f7f7f7;" >
        🔹 Open-source Models
      </td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">FLUX.1 Kontext [Dev]</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.71</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.99</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.88</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">2.19</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.27</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.13</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.98</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.51</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.23</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.18</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Step1X-Edit-v1.2</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.95</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.91</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.04</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">2.68</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.48</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.26</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.90</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.82</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.23</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.22</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Qwen-Image-Edit-2509</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.31</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.34</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.27</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.42</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.73</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.36</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.37</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.91</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.56</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.80</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">FLUX.2 [Dev]</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.35</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.50</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.18</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.83</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.65</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.65</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.31</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.88</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.46</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.70</td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">LongCat-Image-Edit</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.45</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.44</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.53</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.83</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.80</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.60</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.33</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.92</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">3.75</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.82</b></td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;">Qwen-Image-Edit-2511</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.51</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.54</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.57</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.13</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.70</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.46</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.36</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.89</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.16</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.81</u></td>
    </tr><tr>
      <td style="border:1px solid #ddd; padding:8px; text-align:left;"><b>FireRed-Image-Edit</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.56</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.55</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.66</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.34</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.75</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.58</td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.45</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><b>4.97</b></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;"><u>4.07</u></td>
      <td style="border:1px solid #ddd; padding:8px; text-align:center;">4.71</td>
    </tr></tbody>
</table>

### Results on GEdit (official public benchmark)

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Model</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_SC ↑ (EN)</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_PQ ↑ (EN)</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_O ↑ (EN)</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_SC ↑ (CN)</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_PQ ↑ (CN)</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">G_O ↑ (CN)</th>
      </tr>
    </thead>
    <tbody><tr>
        <td colspan="7" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Proprietary Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.396</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.454</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.291</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.540</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.424</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.399</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.0</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.143</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.124</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.701</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.159</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.074</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.692</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana-Pro</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.102</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.344</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>7.738</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.135</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.306</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>7.799</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.5</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.268</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.167</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>7.820</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.254</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.167</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>7.800</strong></td>
      </tr><tr>
        <td colspan="7" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Open-source Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">FLUX.2 [Dev]</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.835</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.064</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.413</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.697</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.046</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.278</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2509</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.974</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.714</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.480</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.988</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.679</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.467</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Step1X-Edit-v1.2</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.974</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.714</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.480</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.988</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.679</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.467</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Longcat-Image-Edit</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.128</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.177</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.748</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.141</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">8.117</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">7.731</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2511</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.297</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.202</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>7.877</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.252</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>8.134</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>7.819</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;"><strong>FireRed-Image-Edit</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.363</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.245</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>7.943</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.287</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>8.227</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>7.887</strong></td>
      </tr></tbody>
  </table>
</div>

### Results on REDEdit-Bench-CN (General Dimensions)

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Model</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Overall</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Add</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Adjust</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">BG</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Beauty</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Color</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Compose</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Extract</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Portrait</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Low-level</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Motion</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Remove</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Replace</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Stylize</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Text</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Viewpoint</th>
      </tr>
    </thead>
    <tbody><tr>
        <td colspan="17" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Proprietary Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.0</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.15</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.55</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.11</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.61</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.83</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.14</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.16</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.48</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.77</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.17</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.68</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.02</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.53</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.94</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.94</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.29</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.5</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.18</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.58</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.09</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.57</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.97</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.05</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.56</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.80</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.99</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.78</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.12</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.53</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.94</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.07</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.53</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.13</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.66</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.26</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.63</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.37</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.13</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.94</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.17</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.83</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.05</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.75</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.07</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.74</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.63</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.69</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.09</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana-Pro</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.48</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.66</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.41</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.58</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.35</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.58</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.36</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.42</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.86</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.46</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.91</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.54</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.79</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.85</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.69</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.75</strong></td>
      </tr><tr>
        <td colspan="17" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Open-source Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2509</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.00</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.45</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.04</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.48</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.36</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.20</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.92</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.64</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.16</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.52</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.66</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.27</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.66</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.81</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.53</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.32</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">FLUX.2 [Dev]</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.05</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.31</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.88</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.57</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.80</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.91</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.85</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.47</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.50</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.43</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.68</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.50</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.47</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.95</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.53</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.88</strong></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Longcat-Image-Edit</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.34</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.25</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.54</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.72</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.92</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.48</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.49</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.31</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.67</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.27</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.61</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.94</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.83</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.30</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2511</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.18</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.50</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.23</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.52</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.61</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.09</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.00</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.22</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.31</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.19</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.66</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.41</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.68</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.83</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.08</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.51</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;"><strong>FireRed-Image-Edit</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.33</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.57</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.37</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.64</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.69</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.45</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.29</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.49</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.50</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.56</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.65</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.47</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.81</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.93</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.49</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.14</td>
      </tr></tbody>
  </table>
</div>

### Results on REDEdit-Bench-EN (General dimensions)

<div style="overflow-x: auto; margin-bottom: 16px;">
  <table style="border-collapse: collapse; width: 100%;">
    <thead>
      <tr>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Model</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Overall</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Add</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Adjust</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">BG</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Beauty</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Color</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Compose</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Extract</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Portrait</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Low-level</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Motion</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Remove</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Replace</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Stylize</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Text</th>
        <th style="padding: 8px; border: 1px solid #d0d7de; background-color: #f6f8fa; white-space: nowrap;">Viewpoint</th>
      </tr>
    </thead>
    <tbody><tr>
        <td colspan="17" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Proprietary Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.15</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.65</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.23</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.60</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.37</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.08</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.98</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.39</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.72</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.03</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.63</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.07</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.68</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.68</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.87</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.23</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.0</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.18</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.59</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.63</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.89</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.10</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.14</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.28</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.77</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.73</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.23</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.56</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.98</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.21</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.42</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Seedream4.5</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.20</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.66</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.08</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.64</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.12</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.07</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.10</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.23</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.74</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.28</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.75</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.24</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.58</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.97</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.20</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.44</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Nano-Banana-Pro</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.42</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.72</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.40</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.64</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.37</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.43</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.32</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.25</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.82</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.36</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.85</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.52</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.75</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.90</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.54</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.51</strong></td>
      </tr><tr>
        <td colspan="17" style="padding: 8px; border: 1px solid #d0d7de; font-weight: 600; text-align: left;">🔹 Open-source Models</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2509</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.99</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.47</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.06</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.49</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.13</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.98</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.85</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.91</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.30</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.71</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.58</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.40</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.67</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.77</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.77</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.85</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">FLUX.2 [Dev]</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.07</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.37</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.96</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.47</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.72</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.86</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.87</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.36</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.44</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.45</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.67</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.02</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.48</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.87</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.80</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.84</strong></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">LongCat-Image-Edit</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.12</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.38</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.04</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.49</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.89</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.10</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.93</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.98</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.47</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.27</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.69</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.24</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.51</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.86</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.83</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.25</td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2511</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.23</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.55</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.17</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.56</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.49</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.07</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.07</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>3.54</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.42</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.52</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.72</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.42</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.65</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.85</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>4.06</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.38</u></td>
      </tr><tr>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: left; white-space: nowrap;"><strong>FireRed-Image-Edit</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.26</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.41</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.33</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.60</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">3.55</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.47</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.25</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><u>3.49</u></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.50</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.44</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">4.65</td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.46</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.70</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.94</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;"><strong>4.44</strong></td>
        <td style="padding: 8px; border: 1px solid #d0d7de; text-align: center; white-space: nowrap;">2.78</td>
      </tr></tbody>
  </table>
</div>

## 📜 License Agreement

The code and the weights of FireRed-Image-Edit are licensed under Apache 2.0. 


## 📝 TODO:
- [x] Release FireRed-Image-Edit-1.0 model.
- [ ] Release REDEdit-Bench, a comprehensive benchmark for image editing evaluation.
- [ ] Release FireRed-Image-Edit-1.0-Distilled model, a distilled version of FireRed-Image-Edit-1.0 for few-step generation.
- [ ] Release FireRed-Image model, a text-to-image generative model.


## 🖊️ Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{firered2026rededit,
      title={FireRed-Image-Edit: A General-Purpose Image Editing Model}, 
      author={Super Intelligence Team},
      year={2026},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
```

## ⚠️ Ethics Statement
FireRed-Image-Edit  has not been specifically designed or comprehensively evaluated for every possible downstream application. Users should be aware of the potential risks and ethical considerations when using this project, and should use it responsibly and in compliance with all applicable laws and regulations.

- **Prohibited Use**: This project must not be used to generate content that is illegal, defamatory, pornographic, harmful, or that violates the privacy, rights, or interests of individuals or organizations.
- **User Responsibility**: Users are solely responsible for any content generated using this project. The authors and contributors assume no responsibility or liability for any misuse of the codebase or for any consequences resulting from its use.



## 🤝 Acknowledgements

We would like to thank the developers of the amazing open-source projects, especially [Qwen-Image](https://github.com/QwenLM/Qwen-Image) for providing a powerful text-to-image foundation model, as well as [Diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## ☎️ Contact

Please contact us and join our Xiaohongshu Group if you have any questions.

#### Xiaohongshu Group 
<img src=assets/xhs.png width="200px">


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FireRedTeam/FireRed-Image-Edit&type=date&legend=top-left)](https://www.star-history.com/#FireRedTeam/FireRed-Image-Edit&type=date&legend=top-left)


---

<div align="center">
  <sub>Built by 小红书智能创作基础技术团队 Xiaohongshu Intelligent Creation Core Technology Team</sub>
</div>