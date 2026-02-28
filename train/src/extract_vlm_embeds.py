"""
使用 QwenVL 从 Jsonl 抽取多模态 embedding，支持分布式与异步保存。
"""
import math
import os
import sys
import requests
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from io import BytesIO

import multiprocessing as mp
import orjson
import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DEVICE = torch.device("cuda", LOCAL_RANK)

ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}

logger.remove()
logger = logger.bind(rank=f"RANK {RANK}")
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{extra[rank]}</green> | <cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | <level>{level}</level> | <level>{message}</level>",
)

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda r: abs(float(r) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def calculate_dimensions(target_area: float, ratio: float) -> Tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height

class JsonlImageDataset(Dataset):
    DEFAULT_SYSTEM_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

    DEFAULT_SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"

    def __init__(
        self,
        jsonl_path: str,
        source_image_column: str = "source_image",
        target_image_column: str = "target_image",
        condition_image_size: int = 384 * 384,
        random_hw_adapt: bool = False,
        image_sample_size: int = 512,
        model_path: Optional[str] = None,
        disable_inverse: bool = False,
        t2i_mode: bool = False,
    ):
        self.df = []
        with open(jsonl_path, "r") as f:
            for idx, line in enumerate(f):
                if idx % WORLD_SIZE != RANK:
                    continue
                self.df.append(orjson.loads(line))

        self.source_image_column = source_image_column
        self.target_image_column = target_image_column
        self.condition_image_size = condition_image_size
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, subfolder="processor")
        self.random_hw_adapt = random_hw_adapt
        self.image_sample_size = image_sample_size
        self.disable_inverse = disable_inverse
        self.t2i_mode = t2i_mode
        if self.t2i_mode:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT_T2I
            assert self.disable_inverse, "T2I mode with inverse instructions is not supported yet."
        else:
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
    def __len__(self):
        return len(self.df)

    def _load_and_resize_image(self, src_image_paths: List[str], tgt_image_path: str) -> Tuple[List[Image.Image], Image.Image, List[Dict], Dict]:
        def _fetch_image(image_path):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
            else:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            return image

        src_images = []
        for src_image_path in src_image_paths:
            src_image = _fetch_image(src_image_path)
            src_images.append(src_image)

        tgt_image = _fetch_image(tgt_image_path)

        source_image_size = [{"width": im.size[0], "height": im.size[1]} for im in src_images]
        edit_image_size = {"width": tgt_image.size[0], "height": tgt_image.size[1]}

        w, h = tgt_image.size
        aspect_ratio_sample_size = {
            k: [x / 512 * self.image_sample_size for x in ASPECT_RATIO_512[k]]
            for k in ASPECT_RATIO_512
        }
        closest_size, _ = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
        closest_size = [int(x / 16) * 16 for x in closest_size]
        closest_size = list(map(int, closest_size))
        if closest_size[0] / h > closest_size[1] / w:
            resize_size = (closest_size[0], int(w * closest_size[0] / h))
        else:
            resize_size = (int(h * closest_size[1] / w), closest_size[1])

        source_transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(closest_size),
        ])

        closest_size_tgt = source_transform(tgt_image)
        closest_size_srcs = [source_transform(im) for im in src_images]

        width, height = closest_size_tgt.size
        aspect_ratio = width / height
        new_width, new_height = calculate_dimensions(self.condition_image_size, aspect_ratio)


        condition_tgt_image = closest_size_tgt.resize((new_width, new_height), Image.Resampling.LANCZOS)
        condition_src_images = [
            im.resize((new_width, new_height), Image.Resampling.LANCZOS) for im in closest_size_srcs
        ]

        return condition_src_images, condition_tgt_image, source_image_size, edit_image_size
    
    def _extract_instructions(self, row: Dict) -> Dict[str, str]:
        return {
            "instruction": row.get("instruction") or "",
            "instruction_cn": row.get("instruction_cn") or "",
            "inverse_instruction": row.get("inverse_instruction") or "",
            "inverse_instruction_cn": row.get("inverse_instruction_cn") or "",
        }

    def _build_conversations(
        self,
        source_images: List[Image.Image],
        target_image: Image.Image,
        instructions: Dict[str, str],
    ) -> List[List[Dict]]:
        if self.t2i_mode:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
            ]
        elif not self.disable_inverse:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
                "",
                instructions["inverse_instruction"],
                instructions["inverse_instruction_cn"],
                "",
            ]
        else:
            instruction_list = [
                instructions["instruction"],
                instructions["instruction_cn"],
                "",
            ]

        conversations = []
        for idx, instruction in enumerate(instruction_list):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": []},
            ]
            condition_images = source_images if idx < 3 else [target_image]
            # if idx >=3 and not self.t2i_mode, then enable inverse, condition_images is [target_image]
            for condition_image in condition_images:
                messages[1]["content"].append({"type": "image", "image": condition_image})
            messages[1]["content"].append({"type": "text", "text": instruction})
            conversations.append(messages)

        return conversations
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df[idx]

        try:
            raw_source = row[self.source_image_column]
            if raw_source is None:
                source_image_paths = []
            elif isinstance(raw_source, str):
                source_image_paths = [raw_source]
            else:
                source_image_paths = list(raw_source)
            source_images, target_image, source_image_size, edit_image_size = self._load_and_resize_image(
                source_image_paths, row[self.target_image_column]
            )
            instructions = self._extract_instructions(row)
            conversations = self._build_conversations(source_images, target_image, instructions)

            model_inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                padding=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            traceback.print_exc()
            return None

        result = row.copy()
        result["_index"] = idx
        result["_jsonl_lineno"] = idx * WORLD_SIZE + RANK
        result["_model_inputs"] = model_inputs
        result["source_image_size"] = source_image_size
        result["edit_image_size"] = edit_image_size
        return result


def collate_fn(batch: List[Dict]) -> List[Dict]:
    return [item for item in batch if item is not None]


class QwenEmbeddingExtractor:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        condition_image_size: int = 384 * 384,
        random_hw_adapt: bool = False,
        image_sample_size: int = 512,
        result_queue: Optional[mp.Queue] = None,
        disable_inverse: bool = False,
        t2i_mode: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.condition_image_size = condition_image_size
        self.model_path = model_path
        self.random_hw_adapt = random_hw_adapt
        self.image_sample_size = image_sample_size
        self.result_queue = result_queue
        self.disable_inverse = disable_inverse
        self.t2i_mode = t2i_mode

        if self.t2i_mode:
            self.prompt_template_start_idx = 34
            assert self.disable_inverse, "T2I mode with inverse instructions is not supported yet."
            self.num_sequences_per_sample = 2
        else:
            self.prompt_template_start_idx = 64
            self.num_sequences_per_sample = 6 if not self.disable_inverse else 3

        logger.info(f"加载模型: {model_path}, 设备: {self.device}, dtype: {self.dtype}")

        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=self.dtype,
            subfolder="text_encoder",
        ).to(self.device)
        self.text_encoder.set_attn_implementation("flash_attention_2")
        self.text_encoder.eval()
        logger.info("模型加载完成")
    
    def _extract_valid_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        valid_hidden_states = []
        for i in range(batch_size):
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_hidden_states.append(hidden_states[i][valid_indices])
        return valid_hidden_states
    
    def _prepare_batch_inputs(
        self,
        batch: List[Dict],
        device: Union[str, torch.device],
    ) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_grid_thw = []

        for item in batch:
            if item is None:
                continue
            model_inputs = item["_model_inputs"]
            num_sequences = model_inputs.input_ids.shape[0]
            for seq_idx in range(num_sequences):
                all_input_ids.append(model_inputs.input_ids[seq_idx])
                all_attention_mask.append(model_inputs.attention_mask[seq_idx])
            if not self.t2i_mode:
                all_pixel_values.append(model_inputs.pixel_values)
                all_image_grid_thw.append(model_inputs.image_grid_thw)

        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=0).to(device)
        attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0).to(device)
        if not self.t2i_mode:
            pixel_values = torch.cat(all_pixel_values, dim=0).to(device)
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0).to(device)
        else:
            pixel_values = None
            image_grid_thw = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    
    def _group_embeddings_by_sample(self, embeddings: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        grouped = []
        current = []
        for emb in embeddings:
            current.append(emb)
            if len(current) == self.num_sequences_per_sample:
                grouped.append(current+[None]*(6-self.num_sequences_per_sample))
                current = []
        return grouped
    
    def _get_qwen_prompt_embeds_batch(
        self,
        batch: List[Dict],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[List[torch.Tensor]]:
        device = device or self.device
        dtype = dtype or self.dtype
        batch_inputs = self._prepare_batch_inputs(batch, device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs["pixel_values"],
                image_grid_thw=batch_inputs["image_grid_thw"],
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        valid_hidden_states = self._extract_valid_hidden_states(
            hidden_states, batch_inputs["attention_mask"]
        )
        start_idx = self.prompt_template_start_idx
        truncated = [h[start_idx:].cpu() for h in valid_hidden_states]
        return self._group_embeddings_by_sample(truncated)
    
    
    def _format_output_result(self, item: Dict, embeddings: List[torch.Tensor]) -> Dict:
        result = {k: v for k, v in item.items() if not k.startswith("_")}
        result["jsonl_lineno"] = item["_jsonl_lineno"]

        result["embeddings_tensor_en"] = embeddings[0]
        result["embeddings_tensor_cn"] = embeddings[1]
        result["embeddings_tensor_droptext"] = embeddings[2]
        result["embeddings_tensor_en_inv"] = embeddings[3]
        result["embeddings_tensor_cn_inv"] = embeddings[4]
        result["embeddings_tensor_droptext_inv"] = embeddings[5]

        return result
    
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        all_embeddings = self._get_qwen_prompt_embeds_batch(batch)
        return [
            self._format_output_result(item, emb_list)
            for item, emb_list in zip(batch, all_embeddings)
        ]

    def run(
        self,
        jsonl_path: str,
        batch_size: int = 512,
        num_workers: int = 4,
    ) -> None:
        logger.info(f"读取 jsonl: {jsonl_path}, batch_size={batch_size}")

        dataset = JsonlImageDataset(
            jsonl_path=jsonl_path,
            source_image_column="source_image",
            target_image_column="edit_image",
            condition_image_size=self.condition_image_size,
            random_hw_adapt=self.random_hw_adapt,
            image_sample_size=self.image_sample_size,
            model_path=self.model_path,
            disable_inverse=self.disable_inverse,
            t2i_mode=self.t2i_mode,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=4,
            pin_memory=(self.device == "cuda"),
        )

        for batch in tqdm(dataloader, desc=f"RANK {RANK} 提取 Embedding"):
            if not batch:
                continue
            try:
                batch_results = self.process_batch(batch)
                if batch_results is not None:
                    self.result_queue.put(batch_results)
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                traceback.print_exc()
                continue

        self.result_queue.put(None)

        logger.info(f"提取完成. 等待保存工作进程退出...")

class SaveWorker(mp.Process):
    def __init__(
        self,
        result_queue: mp.Queue,
        output_jsonl_dir: str,
        embeddings_save_dir: str,
        jsonl_path: str,
    ):
        super().__init__()
        self.result_queue = result_queue
        output_jsonl_dir = Path(output_jsonl_dir)
        output_jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_stem = Path(jsonl_path).stem
        self.out_jsonl = output_jsonl_dir / f"{jsonl_stem}_rank_{RANK:03d}.jsonl"

        embeddings_save_dir = Path(embeddings_save_dir)
        self.embed_dir = embeddings_save_dir / jsonl_stem
        self.embed_dir.mkdir(parents=True, exist_ok=True)

        self.suffixes = ["_en", "_cn", "_droptext", "_en_inv", "_cn_inv", "_droptext_inv"]
        self.tensor_keys = [f"embeddings_tensor{s}" for s in self.suffixes]

    def _save_results(self, results: List[Dict]) -> None:
        if not results:
            return

        with open(self.out_jsonl, "a") as f:
            for rec in results:
                lineno = rec["jsonl_lineno"]
                line_out = {
                    k: v for k, v in rec.items()
                    if k not in self.tensor_keys and k != "jsonl_lineno"
                }
                for suf, key in zip(self.suffixes, self.tensor_keys):
                    tensor = rec.get(key)
                    if tensor is not None:
                        save_path = self.embed_dir / f"{lineno}{suf}.pt"
                        torch.save(tensor, save_path)
                        line_out[key] = save_path.as_posix()

                f.write(orjson.dumps(line_out).decode("utf-8") + "\n")

    def run(self):
        while True:
            try:
                results = self.result_queue.get()
                if results is None:
                    logger.info("SaveWorker 收到结束信号，退出")
                    logger.info(f"处理完成. Embedding 保存到: {self.embed_dir}, 输出 jsonl: {self.out_jsonl}")
                    break
                self._save_results(results)
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                traceback.print_exc()
                time.sleep(0.1)


def main(args) -> None:
    result_queue = mp.Queue(maxsize=2)
    extractor = QwenEmbeddingExtractor(
        model_path=args.model_path,
        device=DEVICE,
        dtype=torch.bfloat16,
        image_sample_size=args.image_sample_size,
        result_queue=result_queue,
        disable_inverse=args.disable_inverse,
        t2i_mode=args.t2i_mode,
    )

    save_worker = SaveWorker(
        result_queue,
        args.output_jsonl_dir,
        args.embeddings_save_dir,
        args.jsonl_path,
    )
    save_worker.start()

    extractor.run(
        jsonl_path=args.jsonl_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    save_worker.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen2VL 从 Jsonl 抽取多模态 embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("jsonl_path", type=str, help="输入 jsonl 路径")
    parser.add_argument("--output_jsonl_dir", type=str, required=True, help="输出 jsonl 目录，各 rank 保存为 {dir}/{输入stem}_rank_{rank}.jsonl")
    parser.add_argument("--embeddings_save_dir", type=str, required=True, help="embedding 保存目录")
    parser.add_argument("--model_path", type=str, default="/dev/shm/Qwen-Image", help="Qwen2VL 模型路径")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--image_sample_size", type=int, default=512, help="图像采样尺寸")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader 进程数")
    parser.add_argument("--disable_inverse", action="store_true", help="禁用逆向提示词")
    parser.add_argument("--t2i_mode", action="store_true", help="T2I 模式")

    main(parser.parse_args())
