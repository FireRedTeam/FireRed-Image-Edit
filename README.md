<p align="center">
    <img src="./assets/logo.png" width="600"/>
<p> 
 <p align="center">
    <a href="https://huggingface.co/FireRedTeam" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FireRedTeam-ffc107?color=ffc107&logoColor=white" style="display: inline-block;"/></a>
    <a href='https://github.com/FireRedTeam/FireRed-Image-Edit'><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
    <a href='https://www.apache.org/licenses/LICENSE-2.0'><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
    <a href="https://arxiv.org/abs/xxxxx" target="_blank"><img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv"></a>
  </p>
<p align="center"> 
    🤗 <a href="https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0">HuggingFace</a> |
    🖥️ <a href="https://huggingface.co/spaces/FireRedTeam/FireRed-Image-Edit-1.0"> Demo</a>
</p>
<p align="center">
    <img src="./assets/teaser.png" width="800"/>
<p> 


## 🔥 FireRed-Image-Edit
**FireRed-Image-Edit** is a general-purpose image editing model that delivers high-fidelity and consistent editing across a wide range of scenarios.

## ✨ Key Features
- **Text Style Preservation**: Maintains text styles with high fidelity, achieving performance comparable to closed-source solutions
- **Photo Restoration**: High-quality old photo restoration and enhancement
- **Multi-Image Editing**: Flexible editing of multiple images such as virtual try-on


## 📰 News
- 2026.02.10: We released FireRed-Image-Edit-1.0. 
## 🎨 Showcase
Some real outputs produced by FireRed-Image-Edit across genearl editing.
<p align="center">
    <img src="./assets/showcase.png" width="800"/>
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
          <span style="white-space: nowrap;">To be released</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image-Edit-1.0-Distilled</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Image-Editing</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">Distilled version of FireRed-Image-Edit-1.0 for faster inference</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">To be released</a></span>
        </td>
      </tr>
      <tr>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">FireRed-Image</td>
        <td style="white-space: nowrap; padding: 8px; border: 1px solid #d0d7de;">Text-to-Image</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">High-quality text-to-image generation model</td>
        <td style="padding: 8px; border: 1px solid #d0d7de;">
          <span style="white-space: nowrap;">To be released</a></span>
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

1. Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```
2. Use the following code snippets to generate or edit images.

### FireRed-Image-Edit-1.0

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests

pipeline = QwenImageEditPlusPipeline.from_pretrained("FireRedTeam/FireRed-Image-Edit-1.0", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("./assets/edit_example1.png").convert("RGB")
prompt = "在书本封面Python的下方，添加一行英文文字2nd Edition"
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_edit.png")
    print("image saved at", os.path.abspath("output_edit.png"))
```


## 📊 Benchmark
To better validate the capabilities of our model, we propose a benchmark called REDEdit-Bench. Our main goal is to build more diverse scenarios and editing instructions that better align with human language, enabling a more comprehensive evaluation of current editing models. We collected over 3,000 images from the internet, and after careful expert-designed selection, we constructed 1,673 bilingual (Chinese–English) editing pairs across 15 categories.

### Inference and Evaluation Code
We provide the inference and evaluation code for REDEdit-Bench. Please refer to the [redbench_infer.py](./src/tools/redbench_infer.py) and [redbench_eval.py](./src/tools/redbench_eval.py) scripts in the `src/tools` directory for more details.

### Benchmark Distribution
The REDEdit-Bench dataset is available at: [REDEdit-Bench Dataset](https://huggingface.co/datasets/FireRedTeam/REDEdit-Bench)

## Results on ImgEdit

| Model | Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | Overall ↑ |
|-------|-----|--------|---------|---------|--------|------------|--------|--------|--------|-----------|
| **🔹 Proprietary Models** | | | | | | | | | | |
| Nano-Banana | **4.62** | 4.41 | 3.68 | 4.34 | 4.39 | **4.40** | 4.18 | **3.72** | **4.83** | 4.29 |
| Seedream4.0 | 4.33 | 4.38 | **3.89** | 4.65 | 4.57 | 4.35 | 4.22 | 3.71 | 4.61 | 4.30 |
| Seedream4.5 | _4.57_ | **4.65** | 2.97 | _4.66_ | 4.46 | 4.37 | _4.92_ | 3.71 | 4.56 | _4.32_ |
| Nano-Banana-Pro | 4.44 | _4.62_ | 3.42 | 4.60 | _4.63_ | 4.32 | **4.97** | 3.64 | 4.69 | **4.37** |
| **🔹 Open-source Models** | | | | | | | | | | |
| FLUX.1 Kontext [Dev] | 3.99 | 3.88 | 2.19 | 4.27 | 3.13 | 3.98 | 4.51 | 3.23 | 4.18 | 3.71 |
| Step1X-Edit-v1.2 | 3.91 | 4.04 | 2.68 | 4.48 | 4.26 | 3.90 | 4.82 | 3.23 | 4.22 | 3.95 |
| Qwen-Image-Edit-2509 | 4.34 | 4.27 | 3.42 | 4.73 | 4.36 | 4.37 | 4.91 | 3.56 | 4.80 | 4.31 |
| FLUX.2 [Dev] | 4.50 | 4.18 | 3.83 | 4.65 | 4.65 | 4.31 | 4.88 | 3.46 | 4.70 | 4.35 |
| LongCat-Image-Edit | 4.44 | 4.53 | 3.83 | _4.80_ | 4.60 | 4.33 | _4.92_ | 3.75 | _4.82_ | 4.45 |
| Qwen-Image-Edit-2511 | 4.54 | _4.57_ | _4.13_ | 4.70 | 4.46 | 4.36 | 4.89 | **4.16** | 4.81 | _4.51_ |
| **FireRed-Image-Edit** | _4.55_ | **4.66** | **4.34** | 4.75 | _4.58_ | _4.45_ | **4.97** | _4.07_ | 4.71 | **4.56** |

### Results on GEdit (official public benchmark)

| Model | G_SC ↑ (EN) | G_PQ ↑ (EN) | G_O ↑ (EN) | G_SC ↑ (CN) | G_PQ ↑ (CN) | G_O ↑ (CN) |
|-------|------------|------------|------------|------------|------------|------------|
| **🔹 Proprietary Models** |||||||
| Nano-Banana | 7.396 | **8.454** | 7.291 | 7.540 | **8.424** | 7.399 |
| Seedream4.0 | _8.143_ | 8.124 | 7.701 | 8.159 | 8.074 | 7.692 |
| Nano-Banana-Pro | 8.102 | _8.344_ | 7.738 | 8.135 | _8.306_ | 7.799 |
| Seedream4.5 | **8.268** | 8.167 | **7.820** | **8.254** | 8.167 | **7.800** |
| **🔹 Open-source Models** |||||||
| FLUX.2 [Dev] | 7.835 | 8.064 | 7.413 | 7.697 | 8.046 | 7.278 |
| Qwen-Image-Edit-2509 | 7.974 | 7.714 | 7.480 | 7.988 | 7.679 | 7.467 |
| Step1X-Edit-v1.2 | 7.974 | 7.714 | 7.480 | 7.988 | 7.679 | 7.467 |
| Longcat-Image-Edit | 8.128 | _8.177_ | 7.748 | 8.141 | 8.117 | 7.731 |
| Qwen-Image-Edit-2511 | _8.297_ | 8.202 | _7.877_ | _8.252_ | 8.134 | _7.819_ |
| **\methodname** | **8.363** | **8.245** | **7.943** | **8.287** | **8.227** | **7.887** |

### Results on REDEdit-Bench-CN (General Dimensions)

| Model | Overall | Add | Adjust | BG | Beauty | Color | Compose | Extract | Portrait | Low-level | Motion | Remove | Replace | Stylize | Text | Viewpoint |
|-------|---------|-----|--------|----|--------|-------|---------|---------|----------|-----------|--------|--------|---------|---------|------|-----------|
| **🔹 Proprietary Models** |||||||||||||||||||
| Seedream4.0 | 4.15 | 4.55 | 4.11 | <u>4.61</u> | 3.83 | 4.14 | <u>4.16</u> | 2.48 | 4.77 | 4.17 | 4.68 | 4.02 | 4.53 | <u>4.94</u> | 3.94 | 3.29 |
| Seedream4.5 | <u>4.18</u> | <u>4.58</u> | 4.09 | 4.57 | 3.97 | 4.12 | 4.05 | 2.56 | <u>4.80</u> | 3.99 | <u>4.78</u> | 4.12 | 4.53 | <u>4.94</u> | 4.07 | <u>3.53</u> |
| Nano-Banana | 4.13 | **4.66** | 4.26 | **4.63** | **4.37** | 4.13 | 3.94 | 3.17 | 4.83 | 4.05 | 4.75 | 4.07 | 4.74 | 3.63 | 3.69 | 3.09 |
| Nano-Banana-Pro | **4.48** | **4.66** | **4.41** | 4.58 | <u>4.35</u> | **4.58** | **4.36** | <u>3.42</u> | **4.86** | **4.46** | **4.91** | **4.54** | **4.79** | 4.85 | **4.69** | **3.75** |
| **🔹 Open-source Models** |||||||||||||||||||
| Qwen-Image-Edit-2509 | 4.00 | 4.45 | 4.04 | 4.48 | 3.36 | 4.20 | 3.92 | 2.64 | 4.16 | 3.52 | 4.66 | 4.27 | 4.66 | 4.81 | 3.53 | 3.32 |
| FLUX.2 [Dev] | 4.05 | 4.31 | 3.88 | 4.57 | **3.80** | 3.91 | 3.85 | 2.47 | 4.50 | <u>4.43</u> | **4.68** | 3.50 | 4.47 | **4.95** | 3.53 | **3.88** |
| Longcat-Image-Edit | 4.12 | 4.34 | 4.25 | 4.54 | <u>3.72</u> | 4.12 | 3.92 | 2.48 | 4.49 | 4.31 | 4.67 | 4.27 | 4.61 | <u>4.94</u> | 3.83 | 3.30 |
| Qwen-Image-Edit-2511 | <u>4.18</u> | <u>4.50</u> | 4.23 | <u>4.52</u> | 3.61 | 4.09 | 4.00 | <u>3.22</u> | 4.31 | 4.19 | <u>4.66</u> | <u>4.41</u> | <u>4.68</u> | 4.83 | 4.08 | <u>3.51</u> |
| **FireRed-Image-Edit** | **4.33** | **4.57** | **4.37** | **4.64** | 3.69 | **4.45** | **4.29** | **3.49** | **4.50** | **4.56** | 4.65 | **4.47** | **4.81** | 4.93 | **4.49** | 3.14 |

### Results on REDEdit-Bench-EN (General dimensions)

| Model | Overall | Add | Adjust | BG | Beauty | Color | Compose | Extract | Portrait | Low-level | Motion | Remove | Replace | Stylize | Text | Viewpoint |
|--------|---------|------|---------|------|---------|--------|----------|----------|-----------|-------------|---------|----------|----------|----------|--------|------------|
| **🔹 Proprietary Models** |||||||||||||||||||
| Nano-Banana | 4.15 | 4.65 | 4.23 | 4.60 | **4.37** | 4.08 | 3.98 | **3.39** | 4.72 | 4.03 | 4.63 | 4.07 | 4.68 | 3.68 | 3.87 | 3.23 |
| Seedream4.0 | 4.18 | <u>4.59</u> | 4.12 | <u>4.63</u> | <u>3.89</u> | 4.10 | <u>4.14</u> | 2.28 | <u>4.77</u> | 4.12 | <u>4.73</u> | 4.23 | 4.56 | **4.98** | 4.21 | 3.42 |
| Seedream4.5 | <u>4.20</u> | 4.66 | 4.08 | 4.64 | 4.12 | 4.07 | 4.10 | 2.23 | 4.74 | <u>4.28</u> | 4.75 | 4.24 | 4.58 | <u>4.97</u> | 4.20 | 3.44 |
| Nano-Banana-Pro | **4.42** | **4.72** | **4.40** | **4.64** | **4.37** | **4.43** | **4.32** | <u>3.25</u> | **4.82** | **4.36** | **4.85** | **4.52** | **4.75** | 4.90 | **4.54** | **3.51** |
| **🔹 Open-source Models** |||||||||||||||||||
| Qwen-Image-Edit-2509 | 3.99 | 4.47 | 4.06 | 4.49 | 3.13 | 3.98 | 3.85 | 2.91 | 4.30 | 3.71 | 4.58 | 4.40 | <u>4.67</u> | 4.77 | 3.77 | 2.85 |
| FLUX.2 [Dev] | 4.07 | 4.37 | 3.96 | 4.47 | 3.72 | 3.86 | 3.87 | 2.36 | 4.44 | <u>4.45</u> | 4.67 | 4.02 | 4.48 | <u>4.87</u> | 3.80 | **3.84** |
| LongCat-Image-Edit | 4.12 | 4.38 | 4.04 | 4.49 | <u>3.89</u> | 4.10 | 3.93 | 2.98 | 4.47 | 4.27 | 4.69 | 4.24 | 4.51 | 4.86 | 3.83 | 3.25 |
| Qwen-Image-Edit-2511 | <u>4.23</u> | <u>4.55</u> | 4.17 | <u>4.56</u> | 3.49 | 4.07 | 4.07 | <u>3.54</u> | 4.42 | **4.52** | <u>4.72</u> | 4.42 | 4.65 | 4.85 | 4.06 | 3.38 |
| **\methodname** | **4.26** | **4.41** | **4.33** | **4.60** | **3.55** | **4.47** | **4.25** | **3.49** | **4.50** | 4.44 | **4.65** | **4.46** | **4.70** | **4.94** | **4.44** | 2.78 |

## 📜 License Agreement

The code and the weights of FireRed-Image-Edit are licensed under Apache 2.0. 


## 📝 TODO:
- [ ] Release FireRed-Image-Edit-1.0 model.
- [ ] Release REDEdit-Bench, a comprehensive benchmark for image editing evaluation.
- [ ] Release FireRed-Image-Edit-1.0-Distilled model, a distilled version of FireRed-Image-Edit-1.0 for few-step generation.
- [ ] Release FireRed-Image model, a text-to-image generative model.


## 🖊️ Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{firered2026,
      title={FireRed-Image-Edit: A General-Purpose Image Editing Model}, 
      author={Changhao Qiao and Chao Hui and Chen Li and Cunzheng Wang and Dejia Song and Jiale Zhang and Jing Li and Qiang Xiang and Runqi Wang and Shuang Sun and Wei Zhu and Xu Tang and Yao Hu and Yibo Chen and Haohua Chen and Haolu Liu and Honghao Cai and Shurui Shi and Shuyang Lin and Sijie Xu and Tianshuo Yuan and Tianze Zhou and Wenxin Yu and Xiangyuan Wang and Xudong Zhou and Yahui Wang and Yandong Guan and Yanqin Chen and Yilian Zhong and Ying Li and Yunhao Bai and Yushun Fang and Zeming Liu and Zhangyu Lai and Zhiqiang Wu},
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

We would like to thank the developers of the amazing open-source projects, including [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co)







