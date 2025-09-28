import argparse
import sys
import os
from typing import List

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core import JanusVQAInference
from config import get_dataset_config, get_output_path


args_dict = {
    "method": "janus_vqa",
    "dataset": "custom",
    "output_dir": "outputs",
    "experiment_suffix": "custom_run",
    "use_visibility_rule": True,
    "use_unanswerable_rule": True,
}
args = argparse.Namespace(**args_dict)


def run_single_experiment(args):
    """Run a single experiment configuration."""
    print(f"Starting experiment: {args.method} on {args.dataset}", flush=True)

    dataset_config = get_dataset_config(args.dataset)

    experiment_suffix = args.experiment_suffix or "default"
    output_path = get_output_path(
        args.output_dir, f"{args.method}_{experiment_suffix}", args.dataset
    )

    print(
        "Initializing model... This may take several minutes on the first run as models are downloaded.",
        flush=True,
    )
    inference = JanusVQAInference(device="auto")
    print("Model initialized successfully.", flush=True)

    print("Running inference on the dataset...", flush=True)
    inference.run_dataset(
        dataset_config,
        output_path,
        use_visibility_rule=args.use_visibility_rule,
        use_unanswerable_rule=args.use_unanswerable_rule,
    )

    print(f"Results saved to: {output_path}", flush=True)


run_single_experiment(args)
