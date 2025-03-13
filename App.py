import os
import sys
import subprocess
import random
import argparse
import time
import tempfile
import json
import gc  # Added for garbage collection

import torch
import gradio as gr
from PIL import Image
import cv2

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download

# ------------------------------
# Helper function for common model files
# ------------------------------
def get_common_file(new_path, old_path):
    """
    Returns the common model file path.
    If the file exists in new_path, then return that,
    else if it exists in old_path, return that (for legacy support).
    """
    if os.path.exists(new_path):
        return new_path
    elif os.path.exists(old_path):
        return old_path
    else:
        print(f"[WARNING] Neither {new_path} nor {old_path} found. Using {old_path} as fallback.")
        return old_path

# ------------------------------
# Config management functions
# ------------------------------
CONFIG_DIR = "configs"
LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config.txt")
DEFAULT_CONFIG_NAME = "Default"

def get_default_config():
    return {
        "model_choice": "WAN 2.1 1.3B (Text/Video-to-Video)",
        "vram_preset": "24GB",
        "aspect_ratio": "16:9",
        "width": 832,
        "height": 480,
        "auto_crop": True,
        "tiled": True,
        "inference_steps": 50,
        "pr_rife": True,
        "pr_rife_multiplier": "2x FPS",
        "cfg_scale": 6.0,
        "sigma_shift": 6.0,
        "num_persistent": "12000000000",
        "torch_dtype": "torch.bfloat16",
        "lora_model": "None",
        "lora_alpha": 1.0,
        "negative_prompt": "still and motionless picture, static",
        "save_prompt": True,
        "multiline": False,
        "num_generations": 1,
        "use_random_seed": True,
        "seed": "",
        "quality": 5,
        "fps": 16,
        "num_frames": 81,
        "denoising_strength": 0.7,
        "tar_lang": "EN",
        "batch_folder": "batch_inputs",
        "batch_output_folder": "batch_outputs",
        "skip_overwrite": True,
        "save_prompt_batch": True,
        # New TeaCache defaults:
        "enable_teacache": False,
        "teacache_thresh": 0.15
    }

if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

default_config = get_default_config()

# Load last used config or create default if missing.
if os.path.exists(LAST_CONFIG_FILE):
    with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
        last_config_name = f.read().strip()
    config_file_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = last_config_name
    else:
        default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_loaded = json.load(f)
            last_config = DEFAULT_CONFIG_NAME
        else:
            config_loaded = default_config
            with open(default_config_path, "w", encoding="utf-8") as f:
                json.dump(config_loaded, f, indent=4)
            last_config = DEFAULT_CONFIG_NAME
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(last_config)
else:
    default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    if os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)
    else:
        config_loaded = default_config
        with open(default_config_path, "w", encoding="utf-8") as f:
            json.dump(config_loaded, f, indent=4)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)

def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return sorted(configs)

def save_config(config_name, model_choice, vram_preset, aspect_ratio, width, height, auto_crop, tiled, inference_steps,
                pr_rife, pr_rife_multiplier, cfg_scale, sigma_shift, num_persistent, torch_dtype, lora_model, lora_alpha,
                negative_prompt, save_prompt, multiline, num_generations, use_random_seed, seed, quality, fps, num_frames,
                denoising_strength, tar_lang, batch_folder, batch_output_folder, skip_overwrite, save_prompt_batch,
                enable_teacache, teacache_thresh):
    if not config_name:
        return "Config name cannot be empty", gr.update(choices=get_config_list())
    config_data = {
        "model_choice": model_choice,
        "vram_preset": vram_preset,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "auto_crop": auto_crop,
        "tiled": tiled,
        "inference_steps": inference_steps,
        "pr_rife": pr_rife,
        "pr_rife_multiplier": pr_rife_multiplier,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "num_persistent": num_persistent,
        "torch_dtype": torch_dtype,
        "lora_model": lora_model,
        "lora_alpha": lora_alpha,
        "negative_prompt": negative_prompt,
        "save_prompt": save_prompt,
        "multiline": multiline,
        "num_generations": num_generations,
        "use_random_seed": use_random_seed,
        "seed": seed,
        "quality": quality,
        "fps": fps,
        "num_frames": num_frames,
        "denoising_strength": denoising_strength,
        "tar_lang": tar_lang,
        "batch_folder": batch_folder,
        "batch_output_folder": batch_output_folder,
        "skip_overwrite": skip_overwrite,
        "save_prompt_batch": save_prompt_batch,
        "enable_teacache": enable_teacache,
        "teacache_thresh": teacache_thresh
    }
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_name)
    return f"Config '{config_name}' saved.", gr.update(choices=get_config_list(), value=config_name)

def load_config(selected_config):
    config_path = os.path.join(CONFIG_DIR, f"{selected_config}.json")
    if not os.path.exists(config_path):
        default_vals = get_default_config()
        return (f"Config '{selected_config}' not found.",
                default_vals["model_choice"],
                default_vals["vram_preset"],
                default_vals["aspect_ratio"],
                default_vals["width"],
                default_vals["height"],
                default_vals["auto_crop"],
                default_vals["tiled"],
                default_vals["inference_steps"],
                default_vals["pr_rife"],
                default_vals["pr_rife_multiplier"],
                default_vals["cfg_scale"],
                default_vals["sigma_shift"],
                default_vals["num_persistent"],
                default_vals["torch_dtype"],
                default_vals["lora_model"],
                default_vals["lora_alpha"],
                default_vals["negative_prompt"],
                default_vals["save_prompt"],
                default_vals["multiline"],
                default_vals["num_generations"],
                default_vals["use_random_seed"],
                default_vals["seed"],
                default_vals["quality"],
                default_vals["fps"],
                default_vals["num_frames"],
                default_vals["denoising_strength"],
                default_vals["tar_lang"],
                default_vals["batch_folder"],
                default_vals["batch_output_folder"],
                default_vals["skip_overwrite"],
                default_vals["save_prompt_batch"],
                "",           # config_name textbox value (reset to empty)
                False,        # enable_teacache (default disabled)
                0.15 )       # teacache_thresh (default)
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(selected_config)
    return (f"Config '{selected_config}' loaded.",
            config_data.get("model_choice", "WAN 2.1 1.3B (Text/Video-to-Video)"),
            config_data.get("vram_preset", "24GB"),
            config_data.get("aspect_ratio", "16:9"),
            config_data.get("width", 832),
            config_data.get("height", 480),
            config_data.get("auto_crop", True),
            config_data.get("tiled", True),
            config_data.get("inference_steps", 50),
            config_data.get("pr_rife", True),
            config_data.get("pr_rife_multiplier", "2x FPS"),
            config_data.get("cfg_scale", 6.0),
            config_data.get("sigma_shift", 6.0),
            config_data.get("num_persistent", "12000000000"),
            config_data.get("torch_dtype", "torch.bfloat16"),
            config_data.get("lora_model", "None"),
            config_data.get("lora_alpha", 1.0),
            config_data.get("negative_prompt", "still and motionless picture, static"),
            config_data.get("save_prompt", True),
            config_data.get("multiline", False),
            config_data.get("num_generations", 1),
            config_data.get("use_random_seed", True),
            config_data.get("seed", ""),
            config_data.get("quality", 5),
            config_data.get("fps", 16),
            config_data.get("num_frames", 81),
            config_data.get("denoising_strength", 0.7),
            config_data.get("tar_lang", "EN"),
            config_data.get("batch_folder", "batch_inputs"),
            config_data.get("batch_output_folder", "batch_outputs"),
            config_data.get("skip_overwrite", True),
            config_data.get("save_prompt_batch", True),
            "",   # config_name textbox (set to empty so user can type a new one)
            config_data.get("enable_teacache", False),
            config_data.get("teacache_thresh", 0.15))

# ------------------------------
# The rest of the app remains largely the same.
# ------------------------------

ASPECT_RATIOS_1_3b = {
    "1:1":  (640, 640),
    "4:3":  (736, 544),
    "3:4":  (544, 736),
    "3:2":  (768, 512),
    "2:3":  (512, 768),
    "16:9": (832, 480),
    "9:16": (480, 832),
    "21:9": (960, 416),
    "9:21": (416, 960),
    "4:5":  (560, 704),
    "5:4":  (704, 560),
}

ASPECT_RATIOS_14b = {
    "1:1":  (960, 960),
    "4:3":  (1104, 832),
    "3:4":  (832, 1104),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "16:9": (1280, 720),
    "16:9_low": (832, 480),
    "9:16": (720, 1280),
    "9:16_low": (480, 832),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}


def update_vram_and_resolution(model_choice, preset):

    print(model_choice)
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
        mapping = {
            "4GB": "0",
            "6GB": "500000000",
            "8GB": "1000000000",
            "10GB": "7000000000",
            "12GB": "7000000000",
            "16GB": "7000000000",
            "24GB": "7000000000",
            "32GB": "7000000000",
            "48GB": "12000000000",
            "80GB": "12000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "3000000000",
            "32GB": "6500000000",
            "48GB": "22000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "32GB": "3500000000",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "1200000000",
            "24GB": "5000000000",
            "32GB": "9500000000",
            "48GB": "20000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
        default_aspect = "16:9"
    else:
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "32GB": "12000000000",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = list(ASPECT_RATIOS_14b.keys())
        default_aspect = "16:9"
    return mapping.get(preset, "12000000000"), resolution_choices, default_aspect


def update_model_settings(model_choice, current_vram_preset):

    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset)
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(default_aspect, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(default_aspect, (1280, 720))
    return (
        gr.update(choices=aspect_options, value=default_aspect),
        default_width,
        default_height,
        num_persistent_val
    )


def update_width_height(aspect_ratio, model_choice):

    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)" or model_choice == "WAN 2.1 14B Image-to-Video 480P":
        default_width, default_height = ASPECT_RATIOS_1_3b.get(aspect_ratio, (832, 480))
    else:
        default_width, default_height = ASPECT_RATIOS_14b.get(aspect_ratio, (1280, 720))
    return default_width, default_height


def update_vram_on_change(preset, model_choice):
    """
    When the VRAM preset changes, update the num_persistent text field based on the current model.
    """
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset)
    return num_persistent_val



def auto_crop_image(image, target_width, target_height):
    """
    Crops and downscales the image to exactly the target resolution.
    The function first crops the image centrally to match the target aspect ratio,
    then resizes it to the target dimensions.
    """
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h

    # Crop the image to the desired aspect ratio.
    if current_ratio > target_ratio:
        # Image is too wide: crop the left and right.
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        # Image is too tall: crop the top and bottom.
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))

    # Resize to the target resolution.
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def auto_crop_video(video_path, target_width, target_height, desired_frame_count, desired_fps=16):
    """
    Reads a video from disk, and for each frame:
      - Downscales if the frame is larger than target dimensions.
      - Performs center crop to get exactly the target resolution.
      - Processes only a number of frames equal to desired_frame_count.
    Saves to a new file (with a _cropped suffix) and returns its path.
    The output video FPS is set to desired_fps.
    The output video duration will be desired_frame_count / desired_fps seconds.
    """
    print(f"[CMD] Starting video processing for file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[CMD] Failed to open video: {video_path}")
        return video_path
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CMD] Original video FPS: {orig_fps}")
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CMD] Original video resolution: {orig_width}x{orig_height}")
    scale = min(1.0, target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    print(f"[CMD] Scaling factor: {scale}. New intermediate resolution: {new_width}x{new_height}")
    
    base, ext = os.path.splitext(video_path)
    out_path = base + "_cropped.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, desired_fps, (target_width, target_height))
    
    frame_count = 0
    while frame_count < desired_frame_count:
        ret, frame = cap.read()
        if not ret:
            print("[CMD] No more frames available from the video.")
            break
        # Downscale if needed.
        if scale < 1.0:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        if w > target_width or h > target_height:
            left = (w - target_width) // 2
            top = (h - target_height) // 2
            frame = frame[top:top+target_height, left:left+target_width]
            print(f"[CMD] Cropped frame {frame_count+1}: left={left}, top={top}")
        else:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            print(f"[CMD] Resized frame {frame_count+1} to target resolution.")
        out.write(frame)
        frame_count += 1
    cap.release()
    out.release()
    print(f"[CMD] Finished processing video. Total frames written: {frame_count}")
    print(f"[CMD] Set output FPS to {desired_fps}. Final video duration: {frame_count/desired_fps:.2f} seconds.")
    print(f"[CMD] Output video saved to: {out_path}")
    return out_path


def prompt_enc(prompt, tar_lang):

    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args

    # Do not clear the WAN pipeline here.
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt

    # Keep prompt_expander in memory for subsequent prompt enhancements.
    return result


def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    enable_teacache, teacache_thresh,
    lora_model, lora_alpha
):

    global loaded_pipeline, loaded_pipeline_config, cancel_flag
    cancel_flag = False  # reset cancellation flag at start
    log_text = ""
    last_used_seed = None
    overall_start_time = time.time()  # overall timer

    improved_videos = []  # List to keep track of each processed video

    # Determine which effective model is used and assign TeaCache model size accordingly.
    if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
        model_choice = "1.3B"
        d = ASPECT_RATIOS_1_3b
        teacache_model_size = "1.3B"
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
        d = ASPECT_RATIOS_14b
        teacache_model_size = "14B"
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        d = ASPECT_RATIOS_14b
        teacache_model_size = "720P"
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 480P":
        model_choice = "14B_image_480p"
        d = ASPECT_RATIOS_1_3b
        teacache_model_size = "480P"
    else:
        return "", "Invalid model choice.", ""
    
    target_width = int(width)
    target_height = int(height)

    # Process video input if applicable.
    if model_choice == "1.3B" and input_video is not None:
        original_video_path = input_video if isinstance(input_video, str) else input_video.name
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_num_frames = min(int(num_frames), total_frames)
            print(f"[CMD] Detected input video frame count: {total_frames}, using effective frame count: {effective_num_frames}")
        else:
            effective_num_frames = int(num_frames)
            print("[CMD] Could not open input video to determine frame count, using provided frame count")
        cap.release()
    else:
        effective_num_frames = int(num_frames)

    # Pre-process based on auto_crop and input type.
    if auto_crop:
        if model_choice == "1.3B" and input_video is not None:
            input_video_path = input_video if isinstance(input_video, str) else input_video.name
            print(f"[CMD] Auto cropping input video: {input_video_path}")
            input_video_path = auto_crop_video(input_video_path, target_width, target_height, effective_num_frames, desired_fps=16)
            input_video = input_video_path
        elif input_image is not None and model_choice in ["14B_image_720p", "14B_image_480p"]:
            # For image-to-video models, keep a copy of the original image to process per iteration.
            original_image = input_image.copy()
    else:
        if input_image is not None:
            original_image = input_image.copy()

    vram_value = num_persistent_input

    # Process LoRA input:
    if lora_model == "None" or not lora_model:
        effective_lora_model = None
    else:
        effective_lora_model = os.path.join("LoRAs", lora_model)

    # Prepare the current configuration dictionary (including LoRA settings).
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": effective_lora_model,
        "lora_alpha": lora_alpha,
    }

    # If the loaded pipeline config has changed (including changes in LoRA selection/scale), reload the pipeline.
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        if effective_lora_model is not None:
            print(f"[CMD] Applying LoRA: {effective_lora_model} with scale {lora_alpha}")
        else:
            print("[CMD] No LoRA selected. Using base model.")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value,
                                              lora_path=effective_lora_model, lora_alpha=lora_alpha)
        loaded_pipeline_config = current_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]

    total_iterations = len(prompts_list) * int(num_generations)
    iteration = 0

    for p in prompts_list:
        for i in range(int(num_generations)):
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user.\n"
                print("[CMD] Generation cancelled by user.")
                duration = time.time() - overall_start_time
                log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "", log_text, str(last_used_seed or "")
            iteration += 1

            # Start timer for this iteration.
            gen_start = time.time()

            log_text += f"[CMD] Generating video {iteration} of {total_iterations} with prompt: {p}\n"
            print(f"[CMD] Generating video {iteration}/{total_iterations} with prompt: {p}")

            # Optionally enhance prompt.
            enhanced_prompt = p

            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                try:
                    current_seed = int(seed_input) if seed_input.strip() != "" else 0
                except Exception as e:
                    current_seed = 0
            last_used_seed = current_seed
            print(f"[CMD] Using resolution: width={target_width}  height={target_height}")

            common_args = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": tiled,
                "width": target_width,
                "height": target_height,
                "num_frames": effective_num_frames,
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
            }

            # Choose branch according to model type.
            if model_choice == "1.3B":
                if input_video is not None:
                    input_video_path = input_video if isinstance(input_video, str) else input_video.name
                    print(f"[CMD] Processing video-to-video with input video: {input_video_path}")
                    video_obj = VideoData(input_video_path, height=target_height, width=target_width)
                    video_data = loaded_pipeline(input_video=video_obj, denoising_strength=denoising_strength,
                                                 enable_teacache=enable_teacache,
                                                 teacache_thresh=teacache_thresh,
                                                 teacache_model_size=teacache_model_size,
                                                 **common_args)
                else:
                    video_data = loaded_pipeline(**common_args,
                                                 enable_teacache=enable_teacache,
                                                 teacache_thresh=teacache_thresh,
                                                 teacache_model_size=teacache_model_size)
                video_filename = get_next_filename(".mp4")
            elif model_choice == "14B_text":
                video_data = loaded_pipeline(**common_args,
                                             enable_teacache=enable_teacache,
                                             teacache_thresh=teacache_thresh,
                                             teacache_model_size=teacache_model_size)
                video_filename = get_next_filename(".mp4")
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if input_image is None:
                    err_msg = "[CMD] Error: Image model selected but no image provided."
                    print(err_msg)
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return "", err_msg, str(last_used_seed or "")
                # For image-to-video, process the image *per iteration* so that we can save the preprocessed image
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image
                video_filename = get_next_filename(".mp4")
                # Save preprocessed image BEFORE generating the video.
                preprocessed_folder = "auto_pre_processed_images"
                if not os.path.exists(preprocessed_folder):
                    os.makedirs(preprocessed_folder)
                base_name = os.path.splitext(os.path.basename(video_filename))[0]
                preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
                processed_image.save(preprocessed_image_filename)
                log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
                print(f"[CMD] Saved auto processed image: {preprocessed_image_filename}")
                video_data = loaded_pipeline(input_image=processed_image,
                                             enable_teacache=enable_teacache,
                                             teacache_thresh=teacache_thresh,
                                             teacache_model_size=teacache_model_size,
                                             **common_args)
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                print(err_msg)
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "", err_msg, str(last_used_seed or "")

            save_video(video_data, video_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved video: {video_filename}\n"
            print(f"[CMD] Saved video: {video_filename}")

            if save_prompt:
                text_filename = os.path.splitext(video_filename)[0] + ".txt"
                generation_details = ""
                generation_details += f"Prompt: {enhanced_prompt}\n"
                generation_details += f"Negative Prompt: {negative_prompt}\n"
                generation_details += f"Used Model: {model_choice_radio}\n"
                generation_details += f"Number of Inference Steps: {inference_steps}\n"
                generation_details += f"Seed: {current_seed}\n"
                generation_details += f"Number of Frames: {effective_num_frames}\n"
                if model_choice == "1.3B" and input_video is not None:
                    generation_details += f"Denoising Strength: {denoising_strength}\n"
                else:
                    generation_details += "Denoising Strength: N/A\n"
                # Log LoRA and TeaCache usage
                if lora_model and lora_model != "None":
                    generation_details += f"LoRA Model: {lora_model} with scale {lora_alpha}\n"
                else:
                    generation_details += "LoRA Model: None\n"
                generation_details += f"TeaCache Enabled: {enable_teacache}\n"
                generation_details += f"TeaCache Threshold: {teacache_thresh}\n"
                generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
                generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
                generation_duration = time.time() - gen_start
                generation_details += f"Generation Duration: {generation_duration:.2f} seconds\n"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
                print(f"[CMD] Saved prompt and parameters: {text_filename}")

            # Apply Practical-RIFE for the current generated video if enabled.
            if pr_rife_enabled and video_filename:
                print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {video_filename}")
                multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                improved_video = os.path.join("outputs", "improved_" + os.path.basename(video_filename))
                model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                cmd = (
                    f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
                    f'--model="{model_dir}" --multi={multiplier_val} '
                    f'--video="{video_filename}" --output="{improved_video}"'
                )
                print(f"[CMD] Running command: {cmd}")
                subprocess.run(cmd, shell=True, check=True, env=os.environ)
                print(f"[CMD] Practical-RIFE finished. Improved video saved to: {improved_video}")
                log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                video_filename = improved_video

            improved_videos.append(video_filename)
            last_video_path = video_filename

    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")

    loaded_pipeline = None
    loaded_pipeline_config = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the last improved video for display along with the log and seed.
    return last_video_path, log_text, str(last_used_seed or "")


def cancel_generation():
    """
    Sets the global cancel flag so that generation loops can end early.
    """
    global cancel_flag
    cancel_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."


def batch_process_videos(
    default_prompt, folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, vram_preset, num_persistent_input,
    torch_dtype, num_frames, inference_steps, aspect_ratio, width, height, auto_crop,
    save_prompt, pr_rife_enabled, pr_rife_radio, lora_model, lora_alpha,
    enable_teacache, teacache_thresh
):

    global loaded_pipeline, loaded_pipeline_config, cancel_batch_flag
    cancel_batch_flag = False  # reset cancellation flag for batch process
    log_text = ""
    
    # For batch processing, we expect only the image-to-video models.
    if model_choice_radio not in ["WAN 2.1 14B Image-to-Video 720P", "WAN 2.1 14B Image-to-Video 480P"]:
        log_text += "[CMD] Batch processing currently only supports the WAN 2.1 14B Image-to-Video models.\n"
        return log_text

    target_width = int(width)
    target_height = int(height)
    
    vram_value = num_persistent_input
    if model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        teacache_model_size = "720P"
    else:  # WAN 2.1 14B Image-to-Video 480P
        model_choice = "14B_image_480p"
        teacache_model_size = "480P"
    
    if lora_model == "None" or not lora_model:
        effective_lora_model = None
    else:
        effective_lora_model = os.path.join("LoRAs", lora_model)
    
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": effective_lora_model,
        "lora_alpha": lora_alpha,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        if effective_lora_model is not None:
            print(f"[CMD] Applying LoRA in batch: {effective_lora_model} with scale {lora_alpha}")
        else:
            print("[CMD] No LoRA selected for batch. Using base model.")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_lora_model, lora_alpha=lora_alpha)
        loaded_pipeline_config = current_config

    common_args_base = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(inference_steps),
        "tiled": True,
        "width": target_width,
        "height": target_height,
        "num_frames": int(num_frames),
    }
    
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        return log_text

    if not os.path.exists(batch_output_folder):
        os.makedirs(batch_output_folder)
        log_text += f"[CMD] Created batch processing outputs folder: {batch_output_folder}\n"

    files = os.listdir(folder_path)
    images = [f for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg"]]
    total_files = len(images)
    log_text += f"[CMD] Found {total_files} image files in folder {folder_path}\n"
    
    for image_file in images:
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            print("[CMD] Batch processing cancelled by user.")
            break

        iter_start = time.time()
        base, ext = os.path.splitext(image_file)
        prompt_path = os.path.join(folder_path, base + ".txt")
        if not os.path.exists(prompt_path):
            log_text += f"[CMD] No prompt txt found for {image_file}, using user entered prompt: {default_prompt}\n"
            print(f"[CMD] No prompt txt found for {image_file}, using user entered prompt: {default_prompt}")
            prompt_content = default_prompt
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            if prompt_content == "":
                log_text += f"[CMD] Prompt txt {base + '.txt'} is empty, using user entered prompt: {default_prompt}\n"
                print(f"[CMD] Prompt txt {base + '.txt'} is empty, using user entered prompt: {default_prompt}")
                prompt_content = default_prompt
            else:
                log_text += f"[CMD] Using user made prompt txt for {image_file}: {prompt_content}\n"
                print(f"[CMD] Using user made prompt txt for {image_file}: {prompt_content}")

        output_filename = os.path.join(batch_output_folder, base + ".mp4")
        if skip_overwrite and os.path.exists(output_filename):
            log_text += f"[CMD] Output video {output_filename} already exists, skipping {image_file} due to skip overwrite.\n"
            print(f"[CMD] Output video {output_filename} already exists, skipping {image_file}.")
            continue

        if use_random_seed:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            try:
                current_seed = int(seed_input) if seed_input.strip() != "" else 0
            except Exception as e:
                current_seed = 0

        common_args = common_args_base.copy()
        common_args["prompt"] = prompt_content
        common_args["seed"] = current_seed

        log_text += f"[CMD] Processing {image_file} with prompt from {base + '.txt'} (or user entered prompt) and seed {current_seed}\n"
        print(f"[CMD] Processing {image_file} with prompt from {base + '.txt'} (or user entered prompt) and seed {current_seed}")
        
        try:
            image_path = os.path.join(folder_path, image_file)
            image_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            log_text += f"[CMD] Failed to open image {image_file}: {str(e)}\n"
            print(f"[CMD] Failed to open image {image_file}: {str(e)}")
            continue

        if auto_crop:
            processed_image = auto_crop_image(image_obj, target_width, target_height)
        else:
            processed_image = image_obj
            
        video_data = loaded_pipeline(input_image=processed_image, **common_args,
                                     enable_teacache=enable_teacache,
                                     teacache_thresh=teacache_thresh,
                                     teacache_model_size=teacache_model_size)
        save_video(video_data, output_filename, fps=fps, quality=quality)
        log_text += f"[CMD] Saved batch generated video: {output_filename}\n"
        print(f"[CMD] Saved batch generated video: {output_filename}")

        # Save the preprocessed image BEFORE video generation is complete.
        if auto_crop:
            preprocessed_folder = "auto_pre_processed_images"
            if not os.path.exists(preprocessed_folder):
                os.makedirs(preprocessed_folder)
            base_name = os.path.splitext(os.path.basename(output_filename))[0]
            preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
            processed_image.save(preprocessed_image_filename)
            log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
            print(f"[CMD] Saved auto processed image: {preprocessed_image_filename}")
        
        generation_duration = time.time() - iter_start
        if save_prompt:
            text_filename = os.path.splitext(output_filename)[0] + ".txt"
            generation_details = ""
            generation_details += f"Prompt: {prompt_content}\n"
            generation_details += f"Negative Prompt: {negative_prompt}\n"
            generation_details += f"Used Model: {model_choice_radio}\n"
            generation_details += f"Number of Inference Steps: {inference_steps}\n"
            generation_details += f"Seed: {current_seed}\n"
            generation_details += f"Number of Frames: {num_frames}\n"
            generation_details += f"Denoising Strength: {denoising_strength}\n"
            if lora_model and lora_model != "None":
                generation_details += f"LoRA Model: {lora_model} with scale {lora_alpha}\n"
            else:
                generation_details += "LoRA Model: None\n"
            generation_details += f"TeaCache Enabled: {enable_teacache}\n"
            generation_details += f"TeaCache Threshold: {teacache_thresh}\n"
            generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
            generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
            generation_details += f"Generation Duration: {generation_duration:.2f} seconds / {(generation_duration/60):.2f} minutes\n"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(generation_details)
            log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
            print(f"[CMD] Saved prompt and parameters: {text_filename}")

        if pr_rife_enabled:
            print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {output_filename}")
            multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
            improved_video = os.path.join(batch_output_folder, "improved_" + os.path.basename(output_filename))
            model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
            cmd = (
                f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
                f'--model="{model_dir}" --multi={multiplier_val} '
                f'--video="{output_filename}" --output="{improved_video}"'
            )
            print(f"[CMD] Running command: {cmd}")
            subprocess.run(cmd, shell=True, check=True, env=os.environ)
            print(f"[CMD] Practical-RIFE finished. Improved video saved to: {improved_video}")
            log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"

    loaded_pipeline = None
    loaded_pipeline_config = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return log_text


def cancel_batch_process():

    global cancel_batch_flag
    cancel_batch_flag = True
    print("[CMD] Batch process cancel button pressed.")
    return "Cancelling batch process..."


def get_next_filename(extension):

    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    existing_files = [f for f in os.listdir(outputs_dir) if f.endswith(extension)]
    current_numbers = []
    for file in existing_files:
        try:
            num = int(os.path.splitext(file)[0])
            current_numbers.append(num)
        except Exception as e:
            continue
    next_number = max(current_numbers, default=0) + 1
    return os.path.join(outputs_dir, f"{next_number:05d}{extension}")


def open_outputs_folder():
    """
    Opens the outputs folder using the default file explorer on Windows or Linux.
    """
    outputs_dir = os.path.abspath("outputs")
    if os.name == 'nt':
        os.startfile(outputs_dir)
    elif os.name == 'posix':
        subprocess.Popen(["xdg-open", outputs_dir])
    else:
        print("[CMD] Opening folder not supported on this OS.")
    return f"Opened {outputs_dir}"


def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):

    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16

    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                os.path.join("models", "Wan-AI", "Wan2.1-T2V-1.3B", "diffusion_pytorch_model.safetensors"),
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_text":
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00001-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00002-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00003-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00004-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00005-of-00006.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-T2V-14B", "diffusion_pytorch_model-00006-of-00006.safetensors")
                ],
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_720p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-720P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                clip_path,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    elif model_choice == "14B_image_480p":
        clip_path = get_common_file(os.path.join("models", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        t5_path = get_common_file(os.path.join("models", "models_t5_umt5-xxl-enc-bf16.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "models_t5_umt5-xxl-enc-bf16.pth"))
        vae_path = get_common_file(os.path.join("models", "Wan2.1_VAE.pth"),
                                  os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "Wan2.1_VAE.pth"))
        model_manager.load_models(
            [
                [
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00001-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00002-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00003-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00004-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00005-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00006-of-00007.safetensors"),
                    os.path.join("models", "Wan-AI", "Wan2.1-I2V-14B-480P", "diffusion_pytorch_model-00007-of-00007.safetensors"),
                ],
                clip_path,
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    
    if lora_path is not None:
        print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
        model_manager.load_lora(lora_path, lora_alpha=lora_alpha)

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)

    if str(num_persistent).strip().lower() == "none":
        num_persistent_val = None
    else:
        try:
            num_persistent_val = int(num_persistent)
        except Exception as e:
            print("[CMD] Warning: could not parse num_persistent_param_in_dit value, defaulting to 6000000000")
            num_persistent_val = 6000000000
    print(f"num_persistent_val {num_persistent_val}")
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    print("[CMD] Model loaded successfully.")
    return pipe


def get_lora_choices():

    lora_folder = "LoRAs"
    if not os.path.exists(lora_folder):
        os.makedirs(lora_folder)
        print("[CMD] 'LoRAs' folder not found. Created 'LoRAs' folder. Please add your LoRA .safetensors files.")
    files = [f for f in os.listdir(lora_folder) if f.endswith(".safetensors")]
    choices = ["None"] + sorted(files)
    return choices


def refresh_lora_list():

    return gr.update(choices=get_lora_choices(), value="None")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Prompt Enhance arguments
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    # Gradio share option
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    args = parser.parse_args()

    # Global pipeline variables
    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    cancel_batch_flag = False
    prompt_expander = None

    with gr.Blocks() as demo:
        gr.Markdown("SECourses Wan 2.1 I2V - V2V - T2V Advanced Gradio APP V30 | Tutorial : https://youtu.be/hnAhveNy-8s | Source : https://www.patreon.com/posts/123105403")
        with gr.Row():
            with gr.Column(scale=4):
                # Model & Resolution settings
                with gr.Row():
                    gr.Markdown("### Model & Resolution")
                with gr.Row():
                    model_choice_radio = gr.Radio(
                        choices=[
                            "WAN 2.1 1.3B (Text/Video-to-Video)",
                            "WAN 2.1 14B Text-to-Video",
                            "WAN 2.1 14B Image-to-Video 720P",
                            "WAN 2.1 14B Image-to-Video 480P"
                        ],
                        label="Model Choice",
                        value=config_loaded.get("model_choice", "WAN 2.1 1.3B (Text/Video-to-Video)")
                    )
                    vram_preset_radio = gr.Radio(
                        choices=["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "32GB", "48GB", "80GB"],
                        label="GPU VRAM Preset",
                        value=config_loaded.get("vram_preset", "24GB")
                    )
                with gr.Row():
                    aspect_ratio_radio = gr.Radio(
                        choices=list(ASPECT_RATIOS_1_3b.keys()),
                        label="Aspect Ratio",
                        value=config_loaded.get("aspect_ratio", "16:9")
                    )
                with gr.Row():
                    width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("width", 832), label="Width")
                    height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("height", 480), label="Height")
                    auto_crop_checkbox = gr.Checkbox(label="Auto Crop", value=config_loaded.get("auto_crop", True))
                    tiled_checkbox = gr.Checkbox(label="Tiled VAE Decode (Disable for 1.3B model for 12GB or more GPUs)", value=config_loaded.get("tiled", True))
                    inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=config_loaded.get("inference_steps", 50), label="Inference Steps")
                gr.Markdown("### Increase Video FPS with Practical-RIFE")
                with gr.Row():
                    pr_rife_checkbox = gr.Checkbox(label="Apply Practical-RIFE", value=config_loaded.get("pr_rife", True))
                    pr_rife_radio = gr.Radio(choices=["2x FPS", "4x FPS"], label="FPS Multiplier", value=config_loaded.get("pr_rife_multiplier", "2x FPS"))
                    cfg_scale_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("cfg_scale", 6.0), label="CFG Scale")
                    sigma_shift_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("sigma_shift", 6.0), label="Sigma Shift")
                gr.Markdown("### GPU Settings")
                with gr.Row():
                    num_persistent_text = gr.Textbox(label="Number of Persistent Parameters In Dit (VRAM)", value=config_loaded.get("num_persistent", "12000000000"))
                    torch_dtype_radio = gr.Radio(
                        choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                        label="Torch DType: float8 (FP8) reduces VRAM and RAM Usage",
                        value=config_loaded.get("torch_dtype", "torch.bfloat16")
                    )
                # TeaCache Settings
                with gr.Row():
                    gr.Markdown("### TeaCache Settings")
                with gr.Row():
                    enable_teacache_checkbox = gr.Checkbox(label="Enable TeaCache (0.1 Threshold for 1.3b model and 0.15 for 14b models recommened - the more faster but lower quality)", value=config_loaded.get("enable_teacache", False))
                    teacache_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=config_loaded.get("teacache_thresh", 0.15), label="Teacache Threshold")
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        label="LoRA Model (Place .safetensors files in 'LoRAs' folder)",
                        choices=get_lora_choices(),
                        value=config_loaded.get("lora_model", "None")
                    )
                    lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha", 1.0), label="LoRA Scale")
                    refresh_lora_button = gr.Button("Refresh LoRAs")
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    cancel_button = gr.Button("Cancel")
                prompt_box = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate", lines=5)
                with gr.Row():
                    tar_lang = gr.Radio(choices=["CH", "EN"], label="Target language for prompt enhance", value=config_loaded.get("tar_lang", "EN"))
                    enhance_button = gr.Button("Prompt Enhance")
                negative_prompt = gr.Textbox(label="Negative Prompt", value=config_loaded.get("negative_prompt", "still and motionless picture, static"), placeholder="Enter negative prompt", lines=2)
                with gr.Row():
                    save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=config_loaded.get("save_prompt", True))
                    multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=config_loaded.get("multiline", False))
                num_generations = gr.Number(label="Number of Generations", value=config_loaded.get("num_generations", 1), precision=0)
                with gr.Row():
                    use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=config_loaded.get("use_random_seed", True))
                    seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value=config_loaded.get("seed", ""))
                with gr.Row():
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("quality", 5), label="Quality")
                    fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=config_loaded.get("fps", 16), label="FPS (for saving video)")
                    num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=config_loaded.get("num_frames", 81), label="Number of Frames")
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                    video_input = gr.Video(label="Input Video (for video-to-video, only for 1.3B)", format="mp4", height=512)
                denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=config_loaded.get("denoising_strength", 0.7),
                                             label="Denoising Strength (only for video-to-video)")
            with gr.Column(scale=3):
                video_output = gr.Video(label="Generated Video", height=720)
                gr.Markdown("### Configuration Management")
                with gr.Row():
                    config_name_textbox = gr.Textbox(label="Config Name (for saving)", placeholder="Enter config name", value="")
                    save_config_button = gr.Button("Save Config")
                with gr.Row():
                    config_dropdown = gr.Dropdown(label="Load Config", choices=get_config_list(), value=last_config)
                    load_config_button = gr.Button("Load Config")
                with gr.Row():
                    config_status = gr.Textbox(label="Config Status", value="", interactive=False, lines=1)
                gr.Markdown("### Batch Image-to-Video Processing")
                batch_folder_input = gr.Textbox(label="Input Folder for Batch Processing", placeholder="Enter input folder path", value=config_loaded.get("batch_folder", "batch_inputs"))
                batch_output_folder_input = gr.Textbox(label="Batch Processing Outputs Folder", placeholder="Enter batch outputs folder path", value=config_loaded.get("batch_output_folder", "batch_outputs"))
                with gr.Row():
                    skip_overwrite_checkbox = gr.Checkbox(label="Skip Overwrite if Output Exists", value=config_loaded.get("skip_overwrite", True))
                    save_prompt_batch_checkbox = gr.Checkbox(label="Save prompt to file (Batch)", value=config_loaded.get("save_prompt_batch", True))
                with gr.Row():
                    batch_process_button = gr.Button("Batch Process")
                    cancel_batch_process_button = gr.Button("Cancel Batch Process")
                batch_status_output = gr.Textbox(label="Batch Process Status Log", lines=10)
                status_output = gr.Textbox(label="Status Log", lines=20)
                last_seed_output = gr.Textbox(label="Last Used Seed", interactive=False)
                open_outputs_button = gr.Button("Open Outputs Folder")

        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider,
                    num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
                    quality_slider, fps_slider,
                    model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
                    num_frames_slider,
                    aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, tiled_checkbox, inference_steps_slider,
                    pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                    enable_teacache_checkbox, teacache_thresh_slider,
                    lora_dropdown, lora_alpha_slider],
            outputs=[video_output, status_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output)
        open_outputs_button.click(fn=open_outputs_folder, outputs=status_output)
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=[prompt_box,
                    batch_folder_input, 
                    batch_output_folder_input, 
                    skip_overwrite_checkbox, 
                    tar_lang, 
                    negative_prompt, 
                    denoising_slider,
                    use_random_seed_checkbox, 
                    seed_input, 
                    quality_slider, 
                    fps_slider, 
                    model_choice_radio, 
                    vram_preset_radio, 
                    num_persistent_text,
                    torch_dtype_radio, 
                    num_frames_slider, 
                    inference_steps_slider,
                    aspect_ratio_radio, 
                    width_slider, 
                    height_slider, 
                    auto_crop_checkbox,
                    save_prompt_batch_checkbox,
                    pr_rife_checkbox, 
                    pr_rife_radio,
                    lora_dropdown,
                    lora_alpha_slider,
                    enable_teacache_checkbox, teacache_thresh_slider],
            outputs=batch_status_output
        )
        cancel_batch_process_button.click(fn=batch_process_videos, outputs=batch_status_output)
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        aspect_ratio_radio.change(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        vram_preset_radio.change(
            fn=update_vram_on_change,
            inputs=[vram_preset_radio, model_choice_radio],
            outputs=num_persistent_text
        )
        refresh_lora_button.click(fn=refresh_lora_list, inputs=[], outputs=lora_dropdown)
        save_config_button.click(
            fn=save_config,
            inputs=[config_name_textbox, model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                    auto_crop_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                    num_persistent_text, torch_dtype_radio, lora_dropdown, lora_alpha_slider, negative_prompt, save_prompt_checkbox, multiline_checkbox,
                    num_generations, use_random_seed_checkbox, seed_input, quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                    batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                    enable_teacache_checkbox, teacache_thresh_slider],
            outputs=[config_status, config_dropdown]
        )
        load_config_button.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[
                config_status, 
                model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio, lora_dropdown, lora_alpha_slider, negative_prompt, save_prompt_checkbox, multiline_checkbox,
                num_generations, use_random_seed_checkbox, seed_input, quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                config_name_textbox,
                enable_teacache_checkbox, teacache_thresh_slider
            ]
        )

        demo.launch(share=args.share, inbrowser=True)