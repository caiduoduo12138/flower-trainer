import argparse
from qwen_image_train import start_training_qwen_image
from qwen_image_2512_train import start_training_qwen_image_2512
from z_image_train import start_training_z_image
from z_image_de_train import start_training_z_image_de
from z_image_base_train import start_training_z_image_base
from qwen_image_edit_2509_train import start_training_qwen_image_edit_2509
from qwen_image_edit_2511_train import start_training_qwen_image_edit_2511


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Start training with given parameters")

    parser.add_argument("--job_type", type=str, default="qwen_image", help="the training job type")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--target_type", type=str, required=True, help="training method in [lora, lokr]")
    parser.add_argument("--concept_sentence", type=str, required=True, help="Trigger word")
    parser.add_argument("--steps", type=int, required=True, help="Number of training steps")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--factor", type=int, default=4, help="LoKr factor")
    parser.add_argument("--low_vram", type=str2bool, default=True, help="Enable low VRAM mode")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--control_folders", type=str, default="[]", help="Path to control dataset folder")
    parser.add_argument("--samples", type=str, default="[]", help="List of sample prompts")

    return parser.parse_args()


def start_job(args):
    assert args.target_type in ["lora", "lokr"], "invalid training method, only support lora or lokr!"
    if args.job_type == "z_image_turbo":
        start_training_z_image(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            samples=args.samples,
        )

    elif args.job_type == "z_image_de_turbo":
        start_training_z_image_de(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            samples=args.samples,
        )

    elif args.job_type == "z_image":
        start_training_z_image_base(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            samples=args.samples,
        )

    elif args.job_type == "qwen_image_2512":
        start_training_qwen_image_2512(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            samples=args.samples,
        )

    elif args.job_type == "qwen_image":
        start_training_qwen_image(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            samples=args.samples,
        )

    elif args.job_type == "qwen_image_edit_2509":
        start_training_qwen_image_edit_2509(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            control_folders=args.control_folders,
            samples=args.samples,
        )

    elif args.job_type == "qwen_image_edit_2511":
        start_training_qwen_image_edit_2511(
            exp_name=args.exp_name,
            target_type=args.target_type,
            concept_sentence=args.concept_sentence,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            factor=args.factor,
            low_vram=args.low_vram,
            dataset_folder=args.dataset_folder,
            control_folders=args.control_folders,
            samples=args.samples,
        )

    else:
        raise RuntimeError("invalid job type, only [z_image_turbo, z_image_de_turbo, z_image_base, qwen_image, qwen_image_edit_2509, qwen_image_edit_2511] are supported!")


if __name__ == "__main__":
    start_job(args=parse_args())

