import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["ALBUMENTATIONS_CHECK_VERSION"] = "0"
import sys
import ast

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

import uuid
import os
import yaml
from slugify import slugify

sys.path.insert(0, "ai-toolkit")
from toolkit.job import get_job


def start_training_z_image_de(
    exp_name: str,
    target_type: str,
    concept_sentence: str,
    steps: int,
    lr: float,
    rank: int,
    factor: int,
    low_vram: bool,
    dataset_folder: str,
    samples: str,
):
    if not exp_name:
        raise RuntimeError("You forgot to insert your LoRA name! This name has to be unique.")
            
    print("Started training")
    slugged_lora_name = slugify(exp_name)

    # Load the default config
    with open("config/examples/run/train_lora_z_image_de_turbo.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["network"]["type"] = target_type
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    config["config"]["process"][0]["train"]["skip_first_sample"] = True
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    config["config"]["process"][0]["network"]["lokr_factor"] = int(factor)
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = False
    config["config"]["process"][0]["save"]["save_every"] = int(steps) // 10
    config["config"]["process"][0]["save"]["max_step_saves_to_keep"] = 10
    config["config"]["process"][0]["sample"]["sample_every"] = int(steps) // 10

    if target_type != "lokr":
        config["config"]["process"][0]["network"]["lokr_factor"] = -1

    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence

    samples = [{"prompt": ii} for ii in ast.literal_eval(samples)]
    if samples:
        config["config"]["process"][0]["train"]["disable_sampling"] = False
        config["config"]["process"][0]["sample"]["samples"] = []
        config["config"]["process"][0]["sample"]["samples"].extend(samples)
    else:
        config["config"]["process"][0]["train"]["disable_sampling"] = True
    
    # Save the updated config
    # generate a random name for the config
    random_config_name = str(uuid.uuid4())
    os.makedirs("tmp", exist_ok=True)
    config_path = f"tmp/{random_config_name}-{slugged_lora_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # run the job locally
    job = get_job(config_path)
    job.run()
    job.cleanup()

    return f"Training completed successfully. Model saved as {slugged_lora_name}"


def main(dataset_path: str):
    start_training_z_image_de("lora_test01", "lora", "zrn", 4800, 1e-4, 16, -1, True, dataset_path, "[]")
    return None


if __name__ == "__main__":
    main("/home/cai/project/ai-toolkit/datasets/zrn")