<p align="center">
    <img src="./assets/logo.png" width="400"/>
<p> 
<p align="center">
  <a href="https://huggingface.co/FireRedTeam" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRedTeam-ffc107?color=ffc107&logoColor=white" style="display: inline-block;"/></a>
  <a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1" target="_blank"><img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRed--Image--Edit--1.1-red" style="display: inline-block;"/></a>
  <a href="https://huggingface.co/spaces/FireRedTeam/FireRed-Image-Edit-1.1" target="_blank"><img alt="HF Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Demo-FireRed--Image--Edit--1.1-red" style="display: inline-block;"/></a>
  <a href="https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="ModelScope Model" src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-FireRed--Image--Edit--1.0-624aff" style="display: inline-block;"/></a>
  <a href="https://modelscope.cn/studios/FireRedTeam/FireRed-Image-Edit-1.0" target="_blank"><img alt="MS Demo" src="https://img.shields.io/badge/%F0%9F%A4%96%20MS%20Demo-FireRed--Image--Edit--1.0-624aff" style="display: inline-block;"/></a>
  <a href='https://github.com/FireRedTeam/FireRed-Image-Edit'><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
  <a href='https://www.apache.org/licenses/LICENSE-2.0'><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://arxiv.org/abs/2602.13344" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv"></a>
</p> 

<p align="center">
    🤗 <a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1">HuggingFace</a> |
    🤖 <a href="https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0">ModelScope</a> |
    🖥️ <a href="https://huggingface.co/spaces/FireRedTeam/FireRed-Image-Edit-1.1">Demo</a> |
    📄 <a href="https://arxiv.org/abs/2602.13344">Technical Report</a>
</p>
<p align="center">
    <img src="./assets/teaser.png" width="800"/>
<p> 


## 🔥 FireRed-Image-Edit
**FireRed-Image-Edit-1.1** Based on the FireRed-Image-Edit-1.0 foundation model, it optimizes portrait consistency, multi-element fusion, stylized text reference, and portrait makeup effects.

**FireRed-Image-Edit-1.0** is a general-purpose image editing model that delivers high-fidelity and consistent editing across a wide range of scenarios. FireRed-Image-Edit delivers leading open-source results with accurate instruction following, high image quality, and consistent visual coherence.

## ✨ Key Features
#### Strong Editing Performance 
- 🆔 **State-of-the-Art Identity Consistency**: Open-source SOTA in character identity preservation, ensuring subjects remain recognizable across complex edits.
- 🧩 **Multi-Element Fusion**: Freely combine 10+ elements with Agent-powered automatic cropping and stitching—no more struggles with short prompts.
- 💄 **Comprehensive Portrait Makeup**: Dozens of styles from professional beauty retouching and yellow/olive skin tone brightening to Halloween witch makeup and creative looks.
- 📝 **Text Style Reference**: Maintains high-fidelity typography and stylized text comparable to closed-source solutions.
- 🖼️ **Professional Photo Restoration**: High-quality old photo repair and enhancement with superior detail recovery.

#### Ultimate Engineering Optimization
- 🔧 **Open LoRA Training Ecosystem:** Full training code released for custom style creation, optimized samplers maximize GPU efficiency for identical tasks, sizes, and input counts.
- ⚡ **Extreme Speed Optimization**: Complete acceleration suite featuring distillation, quantization, and static compilation—delivering 4.5s end-to-end generation with just 30GB VRAM
- 🤖 **Intelligent Agent Workflow**: Automatic multi-image processing handles complex compositions like virtual try-on without requiring lengthy prompt engineering
- 🔌 **Universal Deployment**: Native ComfyUI node support and GGUF lightweight format compatibility for seamless production integration

#### Native Editing Capability from T2I Backbone 
- 🏗️ **Backbone-Agnostic Architecture**: Editing capabilities injected through full Pretrain → SFT → RL pipeline, transferable to any T2I foundation model.

## 📰 News
- 2026.03.03: We release FireRed-Image-Edit-1.1, which, based on the FireRed-Image-Edit-1.0 foundation model, optimizes portrait consistency, multi-element fusion, stylized text reference, and portrait makeup effects.
- 2026.03.01: We offer a lightweight inference script (including distilled [Lora](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0-ComfyUI/blob/main/FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors), quantization, db_cache, and static compilation), now requiring only **30GB VRAM** and **~4.5s/sample**. 🚀 Try it by simply running `python inference.py --optimized True`.
- 2026.02.28: We released the [Train](https://github.com/FireRedTeam/FireRed-Image-Edit/tree/main/train), supporting HSDP/FSDP, Disaggregated Training, and Multi-Condition Aware Bucket Sampler.
- 2026.02.27: We released the [ Agent](#-agent) module for instruction rewriting, multi-image preprocessing, supporting automatic ROI detection, image stitching for editing with more than 3 images.
- 2026.02.27: We provided FireRed-Image-Edit-1.0-**ComfyUI** workflow. Check more details on [Huggingface](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0-ComfyUI)
- 2026.02.14: We released FireRed-Image-Edit-1.0 model weights. Check more details on [Huggingface](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0) and [ModelScope](https://modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0).
- 2026.02.10: We released the [Technical Report](https://arxiv.org/abs/2602.13344) of FireRed-Image-Edit-1.0. 

## 🏆 Evaluation Results
FireRed-Image-Edit establishes a new state-of-the-art among open-source models on Imgedit, Gedit, and RedEdit, while surpassing our closed-source competitors in specific dimensions—a distinction further corroborated by human evaluations highlighting its superior prompt following and visual consistency. Detailed information can be found [here](docs/benchmark.md).
<p align="center">
    <img src="./assets/eval_benchmark.png" width="800"/>
<p>


## 🎨 Showcase
Some real outputs produced by FireRed-Image-Edit across general editing.

### Portrait
<p align="center">
    <img src="./assets/showcase_portrait.png" width="800"/>
<p> 


### Multi-image-fusion
<p align="center">
    <img src="./assets/showcase_multi_fusion.png" width="800"/>
<p> 

### makeup
<p align="center">
    <img src="./assets/showcase_makeup.png" width="800"/>
<p> 


### Text Style Reference
<p align="center">
    <img src="./assets/showcase_mult_text.png" width="800"/>
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
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0-ComfyUI/blob/main/FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors">HuggingFace</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image-Edit-1.1</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image-Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Based on the FireRed-Image-Edit-1.0, it optimizes portrait consistency, multi-element fusion, stylized text reference, and portrait makeup effects.</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">🤗&nbsp;<a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1">HuggingFace</a></span>
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



## 🏋️ How to Train
Training is a **two-step** process:

1. **Extract VLM embeddings** — Run offline extraction on your image–text JSONL.
2. **SFT training** — Train on the extracted embeddings (HSDP/FSDP, multi-node supported).

→ Full details: [train/README.md](train/README.md) (data format, environment, commands).

## 📊 Benchmark
To better validate the capabilities of our model, we propose a benchmark called REDEdit-Bench. Our main goal is to build more diverse scenarios and editing instructions that better align with human language, enabling a more comprehensive evaluation of current editing models. We collected over 3,000 images from the internet, and after careful expert-designed selection, we constructed 1,673 bilingual (Chinese–English) editing pairs across 15 categories.

### Inference and Evaluation Code
We provide the inference and evaluation code for REDEdit-Bench. Please refer to the [redbench_infer.py](./tools/redbench_infer.py) and [redbench_eval.py](./tools/redbench_eval.py) scripts in the `tools` directory for more details.

### Benchmark Distribution
The REDEdit-Bench dataset will be available soon.



<table style="border-collapse: collapse; width: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; font-size: 14px;">
 <thead>
   <tr>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left; background-color: #f6f8fa; white-space: nowrap;">Model</th>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: center; background-color: #f6f8fa; white-space: nowrap;">ImgEdit_O ↑</th>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: center; background-color: #f6f8fa; white-space: nowrap;">GEdit_O ↑ (EN)</th>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: center; background-color: #f6f8fa; white-space: nowrap;">GEdit_O ↑ (CN)</th>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: center; background-color: #f6f8fa; white-space: nowrap;">REDEdit ↑ (EN)</th>
     <th style="border: 1px solid #d0d7de; padding: 8px; text-align: center; background-color: #f6f8fa; white-space: nowrap;">REDEdit ↑ (CN)</th>
   </tr>
 </thead>
 <tbody>
   <!-- Proprietary Models -->
   <tr>
     <td colspan="6" style="border: 1px solid #d0d7de; padding: 8px; font-weight: 700; background-color: #f7f7f7; text-align: left;">
       🔹 Proprietary Models
     </td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Nano-Banana</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.29</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.291</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.399</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.15</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.13</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Seedream4.0</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.30</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.701</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.692</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.18</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.15</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Seedream4.5</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.32</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><strong>7.820</strong></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><strong>7.800</strong></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.20</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.18</u></td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Nano-Banana-Pro</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><strong>4.37</strong></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>7.738</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>7.799</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><strong>4.42</strong></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><strong>4.48</strong></td>
   </tr>
   
   <!-- Open-source Models -->
   <tr>
     <td colspan="6" style="border: 1px solid #d0d7de; padding: 8px; font-weight: 700; background-color: #f7f7f7; text-align: left;">
       🔹 Open-source Models
     </td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Step1X-Edit-v1.2</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">3.95</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.480</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.467</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">—</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">—</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2509</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.31</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.480</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.467</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">3.99</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.00</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">FLUX.2 [Dev]</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.35</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.413</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.278</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.07</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.05</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">LongCat-Image-Edit</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.45</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.748</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">7.731</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.12</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">4.12</td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">Qwen-Image-Edit-2511</td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.51</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>7.877</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>7.819</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.23</u></td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;"><u>4.18</u></td>
   </tr>
   <tr>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: left; white-space: nowrap;">
       <strong>FireRed-Image-Edit</strong>
     </td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">
       <strong>4.56</strong>
     </td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">
       <strong>7.943</strong>
     </td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">
       <strong>7.887</strong>
     </td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">
       <strong>4.26</strong>
     </td>
     <td style="border: 1px solid #d0d7de; padding: 8px; text-align: center;">
       <strong>4.33</strong>
     </td>
   </tr>
 </tbody>
</table>




## 📜 License Agreement

The code and the weights of FireRed-Image-Edit are licensed under Apache 2.0. 


## 📝 TODO:
- [x] Release FireRed-Image-Edit-1.0 model.
- [x] Release FireRed-Image-Edit-1.0-Distilled model, a distilled version of FireRed-Image-Edit-1.0 for few-step generation.
- [x] FireRed-Image-Edit-1.1 released with improvements to portrait consistency, multi-element fusion, stylized text reference, and portrait makeup effects.
- [ ] Release of lightweight editing models comparable to zimage/Flux2-klein and other small-parameter variants.
- [ ] Release REDEdit-Bench, a comprehensive benchmark for image editing evaluation.
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
