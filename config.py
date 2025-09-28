"""
DAM-QA Centralized Configuration

This module contains all dataset, prompt, and experiment configurations for DAM-QA.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

# Base data directory
BASE_DIR = "data"


@dataclass(frozen=True)
class DatasetConfig:
    qa_file: str
    img_folder: str
    max_new_tokens: int
    metric: str
    description: str = ""


# Unified dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "chartqapro_test": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqapro/test.jsonl",
        img_folder=f"{BASE_DIR}/chartqapro/images",
        max_new_tokens=100,
        metric="chartqapro",
        description="ChartQA-Pro test set",
    ),
    "chartqa_test_human": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqa/test_human.jsonl",
        img_folder=f"{BASE_DIR}/chartqa/images",
        max_new_tokens=100,
        metric="relaxed_accuracy",
        description="ChartQA human test set",
    ),
    "chartqa_test_augmented": DatasetConfig(
        qa_file=f"{BASE_DIR}/chartqa/test_augmented.jsonl",
        img_folder=f"{BASE_DIR}/chartqa/images",
        max_new_tokens=100,
        metric="relaxed_accuracy",
        description="ChartQA augmented test set",
    ),
    "docvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/docvqa/val.jsonl",
        img_folder=f"{BASE_DIR}/docvqa/images",
        max_new_tokens=100,
        metric="anls",
        description="DocVQA validation set",
    ),
    "infographicvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/infographicvqa/infographicvqa_val.jsonl",
        img_folder=f"{BASE_DIR}/infographicvqa/images",
        max_new_tokens=100,
        metric="anls",
        description="InfographicVQA validation set",
    ),
    "textvqa_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/textvqa/textvqa_val_updated.jsonl",
        img_folder=f"{BASE_DIR}/textvqa/images",
        max_new_tokens=10,
        metric="vqa_score",
        description="TextVQA validation set",
    ),
    "vqav2_val": DatasetConfig(
        qa_file=f"{BASE_DIR}/vqav2/vqav2_restval.jsonl",
        img_folder=f"{BASE_DIR}/vqav2/images",
        max_new_tokens=10,
        metric="vqa_score",
        description="VQAv2 validation set",
    ),
    "vqav2_restval": DatasetConfig(
        qa_file=f"{BASE_DIR}/vqav2/vqav2_restval.jsonl",
        img_folder=f"{BASE_DIR}/vqav2/images",
        max_new_tokens=10,
        metric="vqa_score",
        description="VQAv2 restval set",
    ),
    "custom": DatasetConfig(
        qa_file=f"{BASE_DIR}/custom/custom_dataset.jsonl",
        img_folder=f"{BASE_DIR}/custom/images",
        max_new_tokens=100,
        metric="vqa_score",
        description="Custom dataset",
    ),
}


# Prompt templates
class PromptTemplates:
    BASE = "Question: {question}\nAnswer:"
    FULL = (
        "<image>\n"
        "Answer each question concisely in a single word or short phrase, "
        "without any lengthy descriptions or explanations.\n"
        "Rely only on information that is clearly visible in the provided image.\n"
        'If the answer cannot be determined from the image, respond with "unanswerable".\n'
        "Question: {question}\nAnswer:"
    )
    DAM_PLANNER = (
        "<image>\n"
        "You are an intelligent visual assistant. Your task is to analyze a user's question about an image and identify the primary object or region of interest mentioned."
        " Your output will be used by an object detection model. Therefore, you must only output a short, descriptive noun phrase for the object."
        " Do not answer the question. Do not explain your reasoning. Be as specific as possible, including colors and other attributes if mentioned.\n\n"
        "Here are some examples:\n"
        'Question: "What is the brand name on the green bottle?"\n'
        'Detection Prompt: "green bottle"\n\n'
        'Question: "How many people are standing near the large clock tower?"\n'
        'Detection Prompt: "large clock tower"\n\n'
        'Question: "In the chart, what does the blue bar represent?"\n'
        'Detection Prompt: "blue bar"\n\n'
        'Question: "{question}"\n'
        "Detection Prompt:"
    )

    @staticmethod
    def get(use_visibility_rule=True, use_unanswerable_rule=True):
        parts = [
            "<image>\n",
            "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n",
        ]
        if use_visibility_rule:
            parts.append(
                "Rely only on information that is clearly visible in the provided image.\n"
            )
        if use_unanswerable_rule:
            parts.append(
                'If the answer cannot be determined from the image, respond with "unanswerable".\n'
            )
        parts.append("Question: {question}\nAnswer:")
        return "".join(parts)


# Granularity and weight configs
GRANULARITY_MODES = {
    "fine": {"window_size": 256, "stride": 128},
    "medium": {"window_size": 512, "stride": 256},
    "coarse": {"window_size": 768, "stride": 384},
}
UNANSWERABLE_WEIGHTS = {
    "zero": 0.0,
    "low": 0.5,
    "normal": 1.0,
    "high": 1.5,
    "very_high": 2.0,
}

# Inference/image params
DEFAULT_INFERENCE_PARAMS = {
    "streaming": False,
    "temperature": 1e-7,
    "top_p": 0.5,
    "num_beams": 1,
}
DEFAULT_IMAGE_PARAMS = {"max_size": 1024}
DEFAULT_WINDOW_PARAMS = {"window_size": 512, "stride": 256}

# Utility functions


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_output_path(output_dir: str, experiment_name: str, dataset_name: str) -> str:
    filename = f"{dataset_name}.csv"
    return os.path.join(output_dir, experiment_name, filename)
