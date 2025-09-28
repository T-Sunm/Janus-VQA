"""
Core inference classes and methods for DAM-QA experiments.
"""

from ast import List
import os
import json
import time
from typing import Optional
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from PIL import Image, ImageDraw
from transformers import AutoModel
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor as DinoProcessor,
)
import sys
import os
from PIL import Image
from torchvision.ops import box_convert
from typing import Dict, Optional, Tuple, List


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .utils import (
    resize_keep_aspect,
    create_full_image_mask,
    get_windows,
    aggregate_votes,
    safe_load_image,
    ensure_dir,
    format_question,
)
from config import (
    DatasetConfig,
    PromptTemplates,
    DEFAULT_INFERENCE_PARAMS,
    DEFAULT_IMAGE_PARAMS,
)


class DAMInference:
    """Base class for DAM model inference."""

    def __init__(
        self, device: str = "auto", model_name: str = "nvidia/DAM-3B-Self-Contained"
    ):
        """
        Initialize DAM model.

        Args:
            device: Device to use ("auto", "cuda:0", "cpu", etc.)
            model_name: Hugging Face model name
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading DAM model on {self.device}...")
        self.dam_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)

        self.dam = self.dam_model.init_dam(
            conv_mode="v1", prompt_mode="full+focal_crop"
        )
        print("DAM model loaded successfully!")

    def predict(
        self,
        img: Image.Image,
        mask: Image.Image,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """
        Generate prediction for image region defined by mask.

        Args:
            img: PIL Image
            mask: PIL mask image
            prompt: Formatted prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        params = DEFAULT_INFERENCE_PARAMS.copy()
        params.update(kwargs)
        params["max_new_tokens"] = max_new_tokens

        tokens = self.dam.get_description(img, mask, prompt, **params)

        if isinstance(tokens, str):
            return tokens.strip()
        else:
            return "".join(tokens).strip()


class GroundingDINOInference:
    """Base class for Grounding DINO."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "IDEA-Research/grounding-dino-tiny",
    ):
        if device == "auto":
            self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gd_processor = DinoProcessor.from_pretrained(model_name)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name
        ).to(self.device)
        print("Grounding DINO model loaded.")

    def predict(self, img: Image.Image, prompt: str) -> Optional[List[float]]:
        """
        Generate predictions for image using Grounding DINO.
        """
        inputs = self.gd_processor(images=img, text=prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.gd_model(**inputs)

        results = self.gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.4, target_sizes=[img.size[::-1]]
        )

        result = results[0]
        if len(result["scores"]) == 0:
            return None

        best_idx = torch.argmax(result["scores"])
        return result["boxes"][best_idx].tolist()


class JanusVQAInference(DAMInference, GroundingDINOInference):
    """
    Inference using JanusVQA two-stage approach:
    1. DAM-Planner generates a text prompt for a region of interest.
    2. Grounding-DINO uses the prompt to find a bounding box.
    3. DAM-VQA answers the question based on the localized region.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize both DAM and Grounding DINO models.
        """
        DAMInference.__init__(
            self, device=device, model_name="nvidia/DAM-3B-Self-Contained"
        )
        GroundingDINOInference.__init__(
            self, device=device, model_name="IDEA-Research/grounding-dino-tiny"
        )
        print("JanusVQA inference ready.")

    def run_dataset(
        self,
        dataset_config: DatasetConfig,
        output_path: str,
        use_visibility_rule: bool = True,
        use_unanswerable_rule: bool = True,
        progress_interval: int = 100,
    ) -> None:
        """
        Run inference on dataset using JanusVQA.
        """
        ensure_dir(output_path)

        prompt_template = PromptTemplates.get(
            use_visibility_rule=use_visibility_rule,
            use_unanswerable_rule=use_unanswerable_rule,
        )

        records = []
        start_time = time.time()

        with open(dataset_config.qa_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(tqdm(fin), start=1):
                entry = json.loads(line)

                qid = entry.get("question_id", idx)
                question = format_question(entry.get("question", ""))
                img_name = entry.get("image") or entry.get("image_id")
                gt = entry.get("answer", [])

                extra_fields = {}
                if "question_type" in entry:
                    extra_fields["question_type"] = entry["question_type"]
                if "year" in entry:
                    extra_fields["year"] = entry["year"]

                img_path = os.path.join(dataset_config.img_folder, img_name)
                img = safe_load_image(img_path)
                if img is None:
                    continue

                img = resize_keep_aspect(img, DEFAULT_IMAGE_PARAMS["max_size"])
                W, H = img.size

                # Stage 1: DAM Planner
                planner_prompt = PromptTemplates.DAM_PLANNER.format(question=question)
                full_mask = create_full_image_mask(W, H)
                detection_prompt = DAMInference.predict(
                    self, img, full_mask, planner_prompt, max_new_tokens=20
                )

                # Stage 2: Grounding DINO
                bbox = GroundingDINOInference.predict(self, img, detection_prompt)

                # Stage 3: DAM VQA on the localized region
                vqa_mask = full_mask
                if bbox:
                    mask = Image.new("L", (W, H), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rectangle(bbox, fill=255)
                    vqa_mask = mask

                vqa_prompt = prompt_template.format(question=question)
                prediction = DAMInference.predict(
                    self, img, vqa_mask, vqa_prompt, dataset_config.max_new_tokens
                )

                record = {
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt,
                    **extra_fields,
                }
                records.append(record)

                if idx % progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {idx} samples in {elapsed:.1f}s")

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        duration = time.time() - start_time
        print(f"Saved {len(records)} rows to {output_path}")
        print(f"Total time: {duration:.1f}s ({duration/60:.1f}m)")


class FullImageInference(DAMInference):
    """Inference using full image without cropping."""

    def run_dataset(
        self,
        dataset_config: DatasetConfig,
        output_path: str,
        use_visibility_rule: bool = True,
        use_unanswerable_rule: bool = True,
        progress_interval: int = 200,
    ) -> None:
        """
        Run inference on entire dataset using full image.

        Args:
            dataset_config: Dataset configuration
            output_path: Output CSV path
            use_visibility_rule: Include visibility constraint in prompt
            use_unanswerable_rule: Include unanswerable instruction in prompt
            progress_interval: Print progress every N samples
        """
        ensure_dir(output_path)

        prompt_template = PromptTemplates.get(
            use_visibility_rule=use_visibility_rule,
            use_unanswerable_rule=use_unanswerable_rule,
        )

        records = []
        start_time = time.time()

        with open(dataset_config.qa_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin, start=1):
                entry = json.loads(line)

                qid = entry.get("question_id", idx)
                question = format_question(entry.get("question", ""))
                img_name = entry.get("image") or entry.get("image_id")
                gt = entry.get("answer", [])

                # Additional fields for some datasets
                extra_fields = {}
                if "question_type" in entry:
                    extra_fields["question_type"] = entry["question_type"]
                if "year" in entry:
                    extra_fields["year"] = entry["year"]

                # Load and process image
                img_path = os.path.join(dataset_config.img_folder, img_name)
                img = safe_load_image(img_path)
                if img is None:
                    continue

                img = resize_keep_aspect(img, DEFAULT_IMAGE_PARAMS["max_size"])
                W, H = img.size

                # Create full image mask and predict
                mask = create_full_image_mask(W, H)
                prompt = prompt_template.format(question=question)
                prediction = self.predict(
                    img, mask, prompt, dataset_config.max_new_tokens
                )

                record = {
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt,
                    **extra_fields,
                }
                records.append(record)

                if idx % progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {idx} samples in {elapsed:.1f}s")

        # Save results
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        duration = time.time() - start_time
        print(f"Saved {len(records)} rows to {output_path}")
        print(f"Total time: {duration:.1f}s ({duration/60:.1f}m)")


class SlidingWindowInference(DAMInference):
    """Inference using sliding window approach with voting."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "nvidia/DAM-3B-Self-Contained",
        window_size: int = 512,
        stride: int = 256,
    ):
        """
        Initialize sliding window inference.

        Args:
            device: Device to use
            model_name: Hugging Face model name
            window_size: Size of sliding windows
            stride: Stride for sliding windows
        """
        super().__init__(device, model_name)
        self.window_size = window_size
        self.stride = stride

    def run_dataset(
        self,
        dataset_config: DatasetConfig,
        output_path: str,
        use_visibility_rule: bool = True,
        use_unanswerable_rule: bool = True,
        unanswerable_weight: float = 1.0,
        progress_interval: int = 100,
    ) -> None:
        """
        Run inference on dataset using sliding window approach.

        Args:
            dataset_config: Dataset configuration
            output_path: Output CSV path
            use_visibility_rule: Include visibility constraint in prompt
            use_unanswerable_rule: Include unanswerable instruction in prompt
            unanswerable_weight: Weight multiplier for unanswerable votes
            progress_interval: Print progress every N samples
        """
        ensure_dir(output_path)

        prompt_template = PromptTemplates.get(
            use_visibility_rule=use_visibility_rule,
            use_unanswerable_rule=use_unanswerable_rule,
        )

        records = []
        start_time = time.time()

        with open(dataset_config.qa_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin, start=1):
                entry = json.loads(line)

                qid = entry.get("question_id", idx)
                question = format_question(entry.get("question", ""))
                img_name = entry.get("image") or entry.get("image_id")
                gt = entry.get("answer", [])

                # Additional fields
                extra_fields = {}
                if "question_type" in entry:
                    extra_fields["question_type"] = entry["question_type"]
                if "year" in entry:
                    extra_fields["year"] = entry["year"]

                # Load and process image
                img_path = os.path.join(dataset_config.img_folder, img_name)
                img = safe_load_image(img_path)
                if img is None:
                    continue

                img = resize_keep_aspect(img, DEFAULT_IMAGE_PARAMS["max_size"])
                W, H = img.size

                votes = defaultdict(float)
                prompt = prompt_template.format(question=question)

                # Full image vote
                mask_full = create_full_image_mask(W, H)
                ans_full = self.predict(
                    img, mask_full, prompt, dataset_config.max_new_tokens
                )
                if ans_full:
                    weight = 1.0
                    if ans_full.lower() == "unanswerable":
                        weight *= unanswerable_weight
                    votes[ans_full] += weight

                # Sliding window votes
                windows = get_windows(W, H, self.window_size, self.stride)
                for x0, y0, x1, y1 in windows:
                    crop = img.crop((x0, y0, x1, y1))
                    mask_crop = Image.new("L", (x1 - x0, y1 - y0), 255)

                    ans = self.predict(
                        crop, mask_crop, prompt, dataset_config.max_new_tokens
                    )
                    if ans:
                        weight = ((x1 - x0) * (y1 - y0)) / (W * H)
                        if ans.lower() == "unanswerable":
                            weight *= unanswerable_weight
                        votes[ans] += weight

                # Aggregate votes
                prediction = aggregate_votes(votes)
                if not prediction:
                    prediction = ans_full

                record = {
                    "question_id": qid,
                    "image_id": img_name,
                    "question": question,
                    "predict": prediction,
                    "gt": gt,
                    **extra_fields,
                }
                records.append(record)

                if idx % progress_interval == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {idx} samples in {elapsed:.1f}s")

        # Save results
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        duration = time.time() - start_time
        print(f"Saved {len(records)} rows to {output_path}")
        print(f"Total time: {duration:.1f}s ({duration/60:.1f}m)")


def run_experiment(
    method: str, dataset: str, output_dir: str, gpu: str = "0", **kwargs
) -> str:
    """
    Run a complete experiment with specified parameters.

    Args:
        method: Inference method ("full_image" or "sliding_window")
        dataset: Dataset name
        output_dir: Output directory
        gpu: GPU device ID
        **kwargs: Additional method-specific parameters

    Returns:
        Path to output CSV file
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # Get dataset config
    from config import get_dataset_config, get_output_path

    dataset_config = get_dataset_config(dataset)

    # Generate output path
    experiment_name = f"{method}_{kwargs.get('experiment_suffix', 'default')}"
    output_path = get_output_path(output_dir, experiment_name, dataset)

    # Initialize inference method
    if method == "full_image":
        inference = FullImageInference(device="auto")
    elif method == "sliding_window":
        window_size = kwargs.get("window_size", 512)
        stride = kwargs.get("stride", 256)
        inference = SlidingWindowInference(
            device="auto", window_size=window_size, stride=stride
        )
    elif method == "janus_vqa":
        inference = JanusVQAInference(device="auto")
    else:
        raise ValueError(f"Unknown method: {method}")

    inference.run_dataset(dataset_config, output_path, **kwargs)

    return output_path
