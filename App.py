import os
import sys
import subprocess
import random
import argparse
import time
import tempfile
import json
import gc
import re
import shutil

import psutil
DEFAULT_CLEAR_CACHE = True if psutil.virtual_memory().total < 31 * 1024**3 else False

import torch
import gradio as gr
from PIL import Image, ImageOps
import cv2

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download

# ------------------------- Utility Functions -------------------------

def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def merge_videos(video_files):
    filelist_path = os.path.join(tempfile.gettempdir(), "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        for vf in video_files:
            f.write(f"file '{os.path.abspath(vf)}'\n")
    merged_video = get_next_filename(".mp4")
    cmd = f'ffmpeg -f concat -safe 0 -i "{filelist_path}" -c copy "{merged_video}"'
    subprocess.run(cmd, shell=True, check=True)
    os.remove(filelist_path)
    return merged_video

def copy_to_outputs(video_path):
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    base = os.path.basename(video_path)
    new_path = os.path.join(outputs_dir, f"input_{base}")
    if not os.path.exists(new_path):
        shutil.copy(video_path, new_path)
    return new_path

def get_common_file(new_path, old_path):
    if os.path.exists(new_path):
        return new_path
    elif os.path.exists(old_path):
        return old_path
    else:
        print(f"[WARNING] Neither {new_path} nor {old_path} found. Using {old_path} as fallback.")
        return old_path

# ------------------------- Pipeline Management Helpers -------------------------

def has_model_config_changed(old_config, new_config):
    keys_to_compare = [
        "model_choice", "torch_dtype", "num_persistent",
        "lora_model", "lora_alpha",
        "lora_model_2", "lora_alpha_2",
        "lora_model_3", "lora_alpha_3",
        "lora_model_4", "lora_alpha_4"
    ]
    for key in keys_to_compare:
        if old_config.get(key) != new_config.get(key):
            return True
    return False

def clear_pipeline_if_needed(pipeline, pipeline_config, new_config):
    global model_manager

    print(f"[CMD - DEBUG] Checking if pipeline needs clearing...")

    if pipeline is not None and has_model_config_changed(pipeline_config, new_config):
        print(f"[CMD - DEBUG] Pipeline config changed. Clearing pipeline.")
        try:
            del pipeline
        except Exception as e:
            print(f"[CMD] Error deleting pipeline: {e}")
        pipeline = None
        pipeline_config = {}

        print(f"[CMD - DEBUG] Checking hasattr(model_manager, 'clear_models'): {hasattr(model_manager, 'clear_models')}")
        print(f"[CMD - DEBUG] dir(model_manager): {dir(model_manager)}")

        if model_manager is not None and hasattr(model_manager, 'clear_models'):
            print(f"[CMD - DEBUG] Calling model_manager.clear_models()")
            model_manager.clear_models()
        else:
            print(f"[CMD - DEBUG] model_manager is None OR hasattr(model_manager, 'clear_models') is False. Skipping model_manager.clear_models()")
        try:
            del model_manager
        except Exception as e:
            print(f"[CMD] Error deleting model_manager: {e}")

        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        print("[CMD] Pipeline and Model Manager cleared due to model config change.")
    else:
        print(f"[CMD - DEBUG] Pipeline config not changed or pipeline is None. No clearing needed.")
    return pipeline, pipeline_config

# ------------------------- Configuration Management -------------------------

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
        "auto_scale": False,
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
        "lora_model_2": "None",
        "lora_alpha_2": 1.0,
        "lora_model_3": "None",
        "lora_alpha_3": 1.0,
        "lora_model_4": "None",
        "lora_alpha_4": 1.0,
        "clear_cache_after_gen": DEFAULT_CLEAR_CACHE,
        "negative_prompt": "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
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
        "enable_teacache": False,
        "tea_cache_l1_thresh": 0.15,
        "tea_cache_model_id": "Wan2.1-T2V-1.3B",
        "extend_factor": 1
    }

if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

default_config = get_default_config()

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

def save_config(config_name, model_choice, vram_preset, aspect_ratio, width, height, auto_crop, auto_scale, tiled, inference_steps,
                pr_rife, pr_rife_multiplier, cfg_scale, sigma_shift, num_persistent, torch_dtype, lora_model, lora_alpha,
                lora_model_2, lora_alpha_2, lora_model_3, lora_alpha_3, lora_model_4, lora_alpha_4, clear_cache_after_gen,
                negative_prompt, save_prompt, multiline, num_generations, use_random_seed, seed, quality, fps, num_frames,
                denoising_strength, tar_lang, batch_folder, batch_output_folder, skip_overwrite, save_prompt_batch,
                enable_teacache, tea_cache_l1_thresh, tea_cache_model_id, extend_factor):
    if not config_name:
        return "Config name cannot be empty", gr.update(choices=get_config_list())
    config_data = {
        "model_choice": model_choice,
        "vram_preset": vram_preset,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "auto_crop": auto_crop,
        "auto_scale": auto_scale,
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
        "lora_model_2": lora_model_2,
        "lora_alpha_2": lora_alpha_2,
        "lora_model_3": lora_model_3,
        "lora_alpha_3": lora_alpha_3,
        "lora_model_4": lora_model_4,
        "lora_alpha_4": lora_alpha_4,
        "clear_cache_after_gen": clear_cache_after_gen,
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
        "tea_cache_l1_thresh": tea_cache_l1_thresh,
        "tea_cache_model_id": tea_cache_model_id,
        "extend_factor": extend_factor
    }
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_name)
    return f"Config '{config_name}' saved.", gr.update(choices=get_config_list(), value=config_name)

def load_config(selected_config):
    config_path = os.path.join(CONFIG_DIR, f"{selected_config}.json")
    default_vals = get_default_config()
    if not os.path.exists(config_path):
        return (
            f"Config '{selected_config}' not found.",
            default_vals["model_choice"],
            default_vals["vram_preset"],
            default_vals["aspect_ratio"],
            default_vals["width"],
            default_vals["height"],
            default_vals["auto_crop"],
            default_vals["auto_scale"],
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
            default_vals["lora_model_2"],
            default_vals["lora_alpha_2"],
            default_vals["lora_model_3"],
            default_vals["lora_alpha_3"],
            default_vals["lora_model_4"],
            default_vals["lora_alpha_4"],
            default_vals["clear_cache_after_gen"],
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
            default_vals["enable_teacache"],
            default_vals["tea_cache_l1_thresh"],
            default_vals["tea_cache_model_id"],
            default_vals["extend_factor"]
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(selected_config)
    return (
        f"Config '{selected_config}' loaded.",
        config_data.get("model_choice", "WAN 2.1 1.3B (Text/Video-to-Video)"),
        config_data.get("vram_preset", "24GB"),
        config_data.get("aspect_ratio", "16:9"),
        config_data.get("width", 832),
        config_data.get("height", 480),
        config_data.get("auto_crop", True),
        config_data.get("auto_scale", False),
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
        config_data.get("lora_model_2", "None"),
        config_data.get("lora_alpha_2", 1.0),
        config_data.get("lora_model_3", "None"),
        config_data.get("lora_alpha_3", 1.0),
        config_data.get("lora_model_4", "None"),
        config_data.get("lora_alpha_4", 1.0),
        config_data.get("clear_cache_after_gen", DEFAULT_CLEAR_CACHE),
        config_data.get("negative_prompt", "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"),
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
        config_data.get("enable_teacache", False),
        config_data.get("tea_cache_l1_thresh", 0.15),
        config_data.get("tea_cache_model_id", "Wan2.1-T2V-1.3B"),
        config_data.get("extend_factor", 1)
    )

def process_random_prompt(prompt):
    pattern = r'<random:\s*([^>]+)>'
    def replacer(match):
        options = [option.strip() for option in match.group(1).split(',') if option.strip()]
        if options:
            return random.choice(options)
        return ''
    return re.sub(pattern, replacer, prompt)

def compute_auto_scale_dimensions(image, default_width, default_height):
    target_area = default_width * default_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return orig_w, orig_h
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return new_w, new_h

def update_target_dimensions(image, auto_scale, current_width, current_height):
    if auto_scale and image is not None:
        try:
            new_w, new_h = compute_auto_scale_dimensions(image, current_width, current_height)
            return new_w, new_h
        except Exception as e:
            return current_width, current_height
    return current_width, current_height

def auto_crop_image(image, target_width, target_height):
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

def auto_scale_image(image, target_width, target_height):
    target_area = target_width * target_height
    orig_w, orig_h = image.size
    if orig_w * orig_h <= target_area:
        return image
    scale_factor = (target_area / (orig_w * orig_h)) ** 0.5
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return image.resize((new_w, new_h), Image.LANCZOS)

def toggle_lora_visibility(current_visibility):
    new_visibility = not current_visibility
    new_label = "Hide More LoRAs" if new_visibility else "Show More LoRAs"
    return gr.update(visible=new_visibility), new_visibility, new_label

def update_tea_cache_model_id(model_choice):
    if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
        return "Wan2.1-T2V-1.3B"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        return "Wan2.1-T2V-14B"
    elif model_choice == "WAN 2.1 14B Image-to-Video 720P":
        return "Wan2.1-I2V-14B-720P"
    elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
        return "Wan2.1-I2V-14B-480P"
    return "Wan2.1-T2V-1.3B"

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

def update_vram_and_resolution(model_choice, preset, torch_dtype):
    print(model_choice)
    if torch_dtype == "torch.float8_e4m3fn":
        if model_choice == "WAN 2.1 14B Text-to-Video":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "0",
                "16GB": "0",
                "24GB": "8,750,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
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
                "24GB": "6,000,000,000",
                "32GB": "15,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "2,500,000,000",
                "16GB": "7,500,000,000",
                "24GB": "15,000,000,000",
                "32GB": "22,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_1_3b.keys())
            default_aspect = "16:9"
        else:
            if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
                mapping = {
                    "4GB": "0",
                    "6GB": "500,000,000",
                    "8GB": "7,000,000,000",
                    "10GB": "7,000,000,000",
                    "12GB": "7,000,000,000",
                    "16GB": "7,000,000,000",
                    "24GB": "7,000,000,000",
                    "32GB": "7,000,000,000",
                    "48GB": "7,000,000,000",
                    "80GB": "7,000,000,000"
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
                    "24GB": "3,000,000,000",
                    "32GB": "6,750,000,000",
                    "48GB": "16,000,000,000",
                    "80GB": "22,000,000,000"
                }
                resolution_choices = list(ASPECT_RATIOS_14b.keys())
                default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect
    else:
        if model_choice == "WAN 2.1 1.3B (Text/Video-to-Video)":
            mapping = {
                "4GB": "0",
                "6GB": "500,000,000",
                "8GB": "7,000,000,000",
                "10GB": "7,000,000,000",
                "12GB": "7,000,000,000",
                "16GB": "7,000,000,000",
                "24GB": "7,000,000,000",
                "32GB": "7,000,000,000",
                "48GB": "7,000,000,000",
                "80GB": "7,000,000,000"
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
                "24GB": "4,250,000,000",
                "32GB": "7,250,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
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
                "24GB": "3,000,000,000",
                "32GB": "5,750,000,000",
                "48GB": "14,500,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        elif model_choice == "WAN 2.1 14B Image-to-Video 480P":
            mapping = {
                "4GB": "0",
                "6GB": "0",
                "8GB": "0",
                "10GB": "0",
                "12GB": "1,500,000,000",
                "16GB": "3,500,000,000",
                "24GB": "8,000,000,000",
                "32GB": "11,000,000,000",
                "48GB": "22,000,000,000",
                "80GB": "22,000,000,000"
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
                "24GB": "3,000,000,000",
                "32GB": "5,750,000,000",
                "48GB": "16,000,000,000",
                "80GB": "22,000,000,000"
            }
            resolution_choices = list(ASPECT_RATIOS_14b.keys())
            default_aspect = "16:9"
        return mapping.get(preset, "12000000000"), resolution_choices, default_aspect

def update_model_settings(model_choice, current_vram_preset, torch_dtype):
    num_persistent_val, aspect_options, default_aspect = update_vram_and_resolution(model_choice, current_vram_preset, torch_dtype)
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
    torch_dtype = config_loaded.get("torch_dtype", "torch.bfloat16")
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset, torch_dtype)
    return num_persistent_val

def prompt_enc(prompt, tar_lang):
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt
    return result

# ------------------------- Detailed Extension Info -------------------------
def show_extension_info():
    info = (
        "**Extended Video Feature – Detailed Explanation:**\n\n"
        "**Purpose:**\n"
        "- The 'Extend Video Factor' slider enables you to automatically lengthen your generated video by appending extra segments. These segments are generated using the last frame of the previous segment to maintain continuity.\n\n"
        "**How It Works:**\n"
        "1. **Initial Generation:**\n"
        "   - If the slider is set to **1×**, no extension is applied and only a single video segment is generated.\n"
        "   - For values greater than **1×**, the app first produces the initial video segment normally.\n\n"
        "2. **Determining the Number of Segments:**\n"
        "   - For **image inputs**: The total number of segments equals the slider value (e.g., 2× produces the base segment plus one extension).\n"
        "   - For **video inputs**: The app copies the original input video to the outputs folder and uses it as the first segment. In this case, the number of additional segments is calculated as: \n"
        "     `additional_segments = extend_factor - 2`  (with 2 segments already present: the copied video and the first generated segment).\n\n"
        "3. **Extension Generation Process:**\n"
        "   - For each additional extension, the app extracts the last frame from the most recent segment using OpenCV.\n"
        "   - This last frame is then re-fed to the generation pipeline using almost the same parameters, ensuring consistency.\n"
        "   - The app may switch the model for the extension based on your original model selection:\n"
        "       - If you used **'WAN 2.1 1.3B (Text/Video-to-Video)'**, then the extension is generated using **'WAN 2.1 14B Image-to-Video 480P'**.\n"
        "       - If you used **'WAN 2.1 14B Text-to-Video'**, then the extension is generated using **'WAN 2.1 14B Image-to-Video 720P'**.\n"
        "       - Otherwise, the same selected model is used for the extension.\n\n"
        "4. **Merging Segments:**\n"
        "   - All generated segments, including the base segment and all extensions, are merged together into one final video using ffmpeg.\n"
        "   - Each segment is also saved individually (with the last frames stored in the 'used_last_frames' folder) for reference.\n\n"
        "5. **Optional Frame-Rate Enhancement:**\n"
        "   - If the Practical-RIFE option is enabled, the final merged video undergoes frame-rate enhancement (doubling or quadrupling the FPS) for smoother motion.\n\n"
        "6. **Batch Processing:**\n"
        "   - When processing a folder of files, a similar extension process is applied. For video files, the last frame is extracted and used to generate additional segments.\n\n"
        "**User Considerations:**\n"
        "- **Continuity & Quality:** The extension relies on the last frame, so slight differences may appear between segments. Adjust the slider for a balance between video length and consistency.\n"
        "- **Resource Usage:** Extra segments use additional GPU/CPU time and memory. Consider lower extension factors if your hardware is constrained.\n"
        "- **Flexibility:** Experiment with various extend factors to suit narrative or scene extension requirements.\n\n"
        "By using the 'Extend Video Factor' and clicking this button, you get full insight into what the app does behind the scenes – from basic video generation to dynamic video extension and optional frame rate enhancement."
    )
    return info

# ------------------------- Pipeline Generation Functions -------------------------

def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, vram_preset, num_persistent_input, torch_dtype, num_frames,
    aspect_ratio, width, height, auto_crop, auto_scale, tiled, inference_steps, pr_rife_enabled, pr_rife_radio, cfg_scale, sigma_shift,
    enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
    lora_model, lora_alpha,
    lora_model_2, lora_alpha_2,
    lora_model_3, lora_alpha_3,
    lora_model_4, lora_alpha_4,
    clear_cache_after_gen, extend_factor
):
    global loaded_pipeline, loaded_pipeline_config, cancel_flag, prompt_expander
    cancel_flag = False
    log_text = ""
    last_used_seed = None
    overall_start_time = time.time()
    final_output_video = None

    input_was_video = False
    orig_video_path = None
    copied_input_video = None
    if input_image is None and input_video is not None:
        input_was_video = True
        orig_video_path = input_video if isinstance(input_video, str) else input_video.name
        if extend_factor > 1:
            copied_input_video = copy_to_outputs(orig_video_path)
            log_text += f"[CMD] Copied input video to: {copied_input_video}\n"

    if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
        model_choice = "1.3B"
        d = ASPECT_RATIOS_1_3b
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
        d = ASPECT_RATIOS_14b
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video 480P":
        model_choice = "14B_image_480p"
        d = ASPECT_RATIOS_1_3b
    else:
        return None, "Invalid model choice.", ""

    target_width = int(width)
    target_height = int(height)

    if model_choice in ["14B_image_720p", "14B_image_480p"]:
        if input_image is None:
            if input_video is not None:
                video_path = input_video if isinstance(input_video, str) else input_video.name
                original_image = extract_last_frame(video_path)
                if original_image is None:
                    err_msg = "[CMD] Error: Could not extract image from provided video. Please upload a valid input image."
                    if clear_cache_after_gen:
                        loaded_pipeline = None
                        loaded_pipeline_config = {}
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return None, err_msg, str(last_used_seed or "")
                log_text += "[CMD] Extracted last frame from input video for image-to-video generation.\n"
            else:
                err_msg = "[CMD] Error: Image model selected but no image provided. Please upload input image."
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")
        else:
            original_image = input_image.copy()
    elif auto_crop or auto_scale:
        if input_image is not None:
            original_image = input_image.copy()
        else:
            original_image = None

    if model_choice == "1.3B" and input_video is not None:
        original_video_path = input_video if isinstance(input_video, str) else input_video.name
        cap = cv2.VideoCapture(original_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_num_frames = min(int(num_frames), total_frames)
            print(f"[CMD] Detected input video frame count: {total_frames}, using effective frame count: {effective_num_frames}")
        else:
            effective_num_frames = int(num_frames)
            print("[CMD] Could not open input video, using provided frame count")
        cap.release()
    else:
        effective_num_frames = int(num_frames)

    num_persistent_input = str(num_persistent_input).replace(",", "")
    vram_value = int(num_persistent_input)
    effective_loras = []
    if lora_model and lora_model != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model), lora_alpha))
    if lora_model_2 and lora_model_2 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_2), lora_alpha_2))
    if lora_model_3 and lora_model_3 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_3), lora_alpha_3))
    if lora_model_4 and lora_model_4 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_4), lora_alpha_4))
        
    new_config = {
         "model_choice": model_choice,
         "torch_dtype": torch_dtype,
         "num_persistent": vram_value,
         "lora_model": lora_model,
         "lora_alpha": lora_alpha,
         "lora_model_2": lora_model_2,
         "lora_alpha_2": lora_alpha_2,
         "lora_model_3": lora_model_3,
         "lora_alpha_3": lora_alpha_3,
         "lora_model_4": lora_model_4,
         "lora_alpha_4": lora_alpha_4
    }
    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, new_config)
    if loaded_pipeline is None:
         loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
         loaded_pipeline_config = new_config

    if multi_line:
        prompts_list = [line.strip() for line in prompt.splitlines() if line.strip()]
    else:
        prompts_list = [prompt.strip()]
    total_iterations = len(prompts_list) * int(num_generations)
    iteration = 0
    generated_segments = []
    last_video_path = None
    for p in prompts_list:
        for i in range(int(num_generations)):
            final_prompt = process_random_prompt(p)
            if cancel_flag:
                log_text += "[CMD] Generation cancelled by user.\n"
                duration = time.time() - overall_start_time
                log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return final_output_video, log_text, str(last_used_seed or "")
            iteration += 1
            gen_start = time.time()
            log_text += f"[CMD] Generating video {iteration} of {total_iterations} with prompt: {final_prompt}\n"
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                try:
                    base_seed = int(seed_input.strip()) if seed_input.strip() != "" else 0
                    current_seed = base_seed + iteration - 1
                except:
                    current_seed = 0
            last_used_seed = current_seed
            print(f"[CMD] Using resolution: width={target_width} height={target_height}")
            common_args = {
                "prompt": final_prompt,
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
            if enable_teacache:
                common_args["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                common_args["tea_cache_model_id"] = tea_cache_model_id
            else:
                common_args["tea_cache_l1_thresh"] = None
                common_args["tea_cache_model_id"] = ""
            if model_choice == "1.3B":
                if input_video is not None:
                    input_video_path = input_video if isinstance(input_video, str) else input_video.name
                    print(f"[CMD] Processing video-to-video with input video: {input_video_path}")
                    video_obj = VideoData(input_video_path, height=target_height, width=target_width)
                    video_data = loaded_pipeline(
                        input_video=video_obj,
                        denoising_strength=denoising_strength,
                        **common_args,
                        cancel_fn=lambda: cancel_flag
                    )
                    if cancel_flag:
                        log_text += "[CMD] Generation cancelled by user mid-run.\n"
                        duration = time.time() - overall_start_time
                        log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                        log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                        if clear_cache_after_gen:
                            loaded_pipeline = None
                            loaded_pipeline_config = {}
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        return final_output_video, log_text, str(last_used_seed or "")
                else:
                    video_data = loaded_pipeline(
                        **common_args,
                        cancel_fn=lambda: cancel_flag
                    )
                    if cancel_flag:
                        log_text += "[CMD] Generation cancelled by user mid-run.\n"
                        duration = time.time() - overall_start_time
                        log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                        log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                        if clear_cache_after_gen:
                            loaded_pipeline = None
                            loaded_pipeline_config = {}
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        return final_output_video, log_text, str(last_used_seed or "")
                video_filename = get_next_filename(".mp4")
            elif model_choice == "14B_text":
                video_data = loaded_pipeline(
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user mid-run.\n"
                    duration = time.time() - overall_start_time
                    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                    log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                    if clear_cache_after_gen:
                        loaded_pipeline = None
                        loaded_pipeline_config = {}
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return final_output_video, log_text, str(last_used_seed or "")
                video_filename = get_next_filename(".mp4")
            elif model_choice in ["14B_image_720p", "14B_image_480p"]:
                if auto_crop:
                    processed_image = auto_crop_image(original_image, target_width, target_height)
                elif auto_scale:
                    processed_image = auto_scale_image(original_image, target_width, target_height)
                else:
                    processed_image = original_image
                video_filename = get_next_filename(".mp4")
                preprocessed_folder = "auto_pre_processed_images"
                if not os.path.exists(preprocessed_folder):
                    os.makedirs(preprocessed_folder)
                base_name = os.path.splitext(os.path.basename(video_filename))[0]
                preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
                processed_image.save(preprocessed_image_filename)
                log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
                video_data = loaded_pipeline(
                    input_image=processed_image,
                    **common_args,
                    cancel_fn=lambda: cancel_flag
                )
                if cancel_flag:
                    log_text += "[CMD] Generation cancelled by user mid-run.\n"
                    duration = time.time() - overall_start_time
                    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                    log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                    if clear_cache_after_gen:
                        loaded_pipeline = None
                        loaded_pipeline_config = {}
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    return final_output_video, log_text, str(last_used_seed or "")
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                if clear_cache_after_gen:
                    loaded_pipeline = None
                    loaded_pipeline_config = {}
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return None, err_msg, str(last_used_seed or "")
            save_video(video_data, video_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved video: {video_filename}\n"
            final_output_video = video_filename
            if save_prompt:
                text_filename = os.path.splitext(video_filename)[0] + ".txt"
                generation_details = ""
                generation_details += f"Prompt: {final_prompt}\n"
                generation_details += f"Negative Prompt: {negative_prompt}\n"
                generation_details += f"Used Model: {model_choice_radio}\n"
                generation_details += f"Number of Inference Steps: {inference_steps}\n"
                generation_details += f"CFG Scale: {cfg_scale}\n"
                generation_details += f"Sigma Shift: {sigma_shift}\n"
                generation_details += f"Seed: {current_seed}\n"
                generation_details += f"Number of Frames: {effective_num_frames}\n"
                if model_choice == "1.3B" and input_video is not None:
                    generation_details += f"Denoising Strength: {denoising_strength}\n"
                else:
                    generation_details += "Denoising Strength: N/A\n"
                if effective_loras:
                    lora_details = ", ".join([f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras])
                    generation_details += f"LoRA Models: {lora_details}\n"
                else:
                    generation_details += "LoRA Model: None\n"
                generation_details += f"TeaCache Enabled: {enable_teacache}\n"
                if enable_teacache:
                    generation_details += f"TeaCache L1 Threshold: {tea_cache_l1_thresh}\n"
                    generation_details += f"TeaCache Model ID: {tea_cache_model_id}\n"
                generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
                generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
                generation_details += f"Final Resolution: {target_width}x{target_height}\n"
                generation_duration = time.time() - gen_start
                generation_details += f"Generation Duration: {generation_duration:.2f} seconds\n"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(generation_details)
                log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
            # Apply Practical-RIFE enhancement for the generated segment if enabled and FPS <= 29.
            if pr_rife_enabled and video_filename:
                cap = cv2.VideoCapture(video_filename)
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if source_fps <= 29:
                    print(f"[CMD] Applying Practical-RIFE with multiplier {pr_rife_radio} on video {video_filename}")
                    multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                    improved_video = os.path.join("outputs", "improved_" + os.path.basename(video_filename))
                    model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                    cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{model_dir}" --multi={multiplier_val} --video="{video_filename}" --output="{improved_video}"'
                    subprocess.run(cmd, shell=True, check=True, env=os.environ)
                    log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                    video_filename = improved_video
                    final_output_video = improved_video
                else:
                    log_text += f"[CMD] Skipping Practical-RIFE because source video FPS ({source_fps:.2f}) is above 29.\n"
            last_video_path = video_filename
            generated_segments.append(last_video_path)
            
    # --- New extend video logic using the same global pipeline ---
    if extend_factor > 1 and last_video_path:
        if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
            ext_model_radio = "WAN 2.1 14B Image-to-Video 480P"
        elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
            ext_model_radio = "WAN 2.1 14B Image-to-Video 720P"
        else:
            ext_model_radio = model_choice_radio
        if ext_model_radio == "WAN 2.1 14B Image-to-Video 480P":
            ext_model_code = "14B_image_480p"
        elif ext_model_radio == "WAN 2.1 14B Image-to-Video 720P":
            ext_model_code = "14B_image_720p"
        else:
            ext_model_code = model_choice

        if input_was_video and copied_input_video is not None:
            segments = [copied_input_video]
            segments.append(last_video_path)
            log_text += f"[CMD] Starting extension with input video: {copied_input_video}\n"
            cur_last_video = last_video_path
        else:
            segments = [last_video_path]
            cur_last_video = last_video_path
            
        additional_extensions = int(extend_factor) - (2 if input_was_video else 1)
        
        ext_config = {"model_choice": ext_model_code, "torch_dtype": torch_dtype, "num_persistent": vram_value}
        loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, ext_config)
        if loaded_pipeline is None:
            loaded_pipeline = load_wan_pipeline(ext_model_code, torch_dtype, vram_value, lora_path=[], lora_alpha=None)
            loaded_pipeline_config = ext_config
            
        for ext_iter in range(1, additional_extensions + 1):
            last_frame = extract_last_frame(cur_last_video)
            if last_frame is None:
                log_text += f"[CMD] Failed to extract last frame from {cur_last_video} for extension {ext_iter}.\n"
                break
                
            used_folder = "used_last_frames"
            if not os.path.exists(used_folder):
                os.makedirs(used_folder)
            base_name = os.path.splitext(os.path.basename(cur_last_video))[0]
            last_frame_filename = os.path.join(used_folder, f"{base_name}_lastframe.png")
            last_frame.save(last_frame_filename)
            log_text += f"[CMD] Saved used last frame: {last_frame_filename}\n"
            
            new_width, new_height = last_frame.size
            common_args_ext = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": random.randint(0, 2**32 - 1) if use_random_seed else int(seed_input.strip() or 0) + ext_iter,
                "tiled": tiled,
                "width": new_width,
                "height": new_height,
                "num_frames": int(num_frames),
                "cfg_scale": cfg_scale,
                "sigma_shift": sigma_shift,
            }
            if enable_teacache:
                common_args_ext["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                common_args_ext["tea_cache_model_id"] = tea_cache_model_id
            else:
                common_args_ext["tea_cache_l1_thresh"] = None
                common_args_ext["tea_cache_model_id"] = ""
                
            log_text += f"[CMD] Generating extension segment {ext_iter} using model {ext_model_radio}\n"
            video_filename_ext = get_next_filename(".mp4")
            video_data_ext = loaded_pipeline(
                input_image=last_frame,
                **common_args_ext,
                cancel_fn=lambda: cancel_flag
            )
            
            if cancel_flag:
                log_text += "[CMD] Extension generation cancelled by user mid-run.\n"
                break
                
            save_video(video_data_ext, video_filename_ext, fps=fps, quality=quality)
            log_text += f"[CMD] Saved extension segment {ext_iter} video: {video_filename_ext}\n"
            segments.append(video_filename_ext)
            cur_last_video = video_filename_ext
            
        if len(segments) > 1:
            try:
                merged_video = merge_videos(segments)
                log_text += f"[CMD] Merged extended video saved as: {merged_video}\n"
                last_video_path = merged_video
                final_output_video = merged_video
            except Exception as e:
                log_text += f"[CMD] Error merging videos: {str(e)}\n"
                last_video_path = segments[-1]
                final_output_video = segments[-1]
        
        # Apply RIFE on the extended merged video if enabled and FPS <= 29.
        if pr_rife_enabled and final_output_video:
            cap = cv2.VideoCapture(final_output_video)
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if source_fps <= 29:
                multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                improved_video = os.path.join("outputs", "improved_" + os.path.basename(final_output_video))
                model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                cmd = f'"{sys.executable}" "Practical-RIFE/inference_video.py" --model="{model_dir}" --multi={multiplier_val} --video="{final_output_video}" --output="{improved_video}"'
                subprocess.run(cmd, shell=True, check=True, env=os.environ)
                log_text += f"[CMD] Applied Practical-RIFE to extended video with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                final_output_video = improved_video
            else:
                log_text += f"[CMD] Skipping Practical-RIFE on extended video because source video FPS ({source_fps:.2f}) is above 29.\n"
            
    overall_duration = time.time() - overall_start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes). Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Overall Duration: {overall_duration:.2f} seconds. Last used seed: {last_used_seed}")
    
    if clear_cache_after_gen:
        loaded_pipeline = None
        loaded_pipeline_config = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if final_output_video and os.path.exists(final_output_video):
        return final_output_video, log_text, str(last_used_seed or "")
    elif last_video_path and os.path.exists(last_video_path):
        return last_video_path, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")
    else:
        return None, log_text + "\n[CMD] Warning: Could not generate valid output video.", str(last_used_seed or "")

def cancel_generation():
    global cancel_flag
    cancel_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."

def batch_process_videos(
    default_prompt, folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, vram_preset, num_persistent_input,
    torch_dtype, num_frames, inference_steps, aspect_ratio, width, height, auto_crop, auto_scale,
    tiled, cfg_scale, sigma_shift,
    save_prompt, pr_rife_enabled, pr_rife_radio, lora_model, lora_alpha,
    lora_model_2, lora_alpha_2,
    lora_model_3, lora_alpha_3,
    lora_model_4, lora_alpha_4,
    enable_teacache, tea_cache_l1_thresh, tea_cache_model_id,
    clear_cache_after_gen, extend_factor
):
    global loaded_pipeline, loaded_pipeline_config, cancel_batch_flag
    cancel_batch_flag = False
    log_text = ""
    if model_choice_radio not in ["WAN 2.1 14B Image-to-Video 720P", "WAN 2.1 14B Image-to-Video 480P"]:
        log_text += f"[CMD] Batch processing currently only supports the WAN 2.1 14B Image-to-Video models.\n"
        return log_text
    target_width = int(width)
    target_height = int(height)
    num_persistent_input = str(num_persistent_input).replace(",", "")
    vram_value = int(num_persistent_input)
    if model_choice_radio == "WAN 2.1 14B Image-to-Video 720P":
        model_choice = "14B_image_720p"
    else:
        model_choice = "14B_image_480p"
    effective_loras = []
    if lora_model and lora_model != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model), lora_alpha))
    if lora_model_2 and lora_model_2 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_2), lora_alpha_2))
    if lora_model_3 and lora_model_3 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_3), lora_alpha_3))
    if lora_model_4 and lora_model_4 != "None":
        effective_loras.append((os.path.join("LoRAs", lora_model_4), lora_alpha_4))
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
        "lora_model": lora_model,
        "lora_alpha": lora_alpha,
        "lora_model_2": lora_model_2,
        "lora_alpha_2": lora_alpha_2,
        "lora_model_3": lora_model_3,
        "lora_alpha_3": lora_alpha_3,
        "lora_model_4": lora_model_4,
        "lora_alpha_4": lora_alpha_4
    }
    if cancel_batch_flag:
        log_text += "[CMD] Batch processing cancelled before model loading.\n"
        cancel_batch_flag = False
        return log_text
    loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, current_config)
    if loaded_pipeline is None:
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value, lora_path=effective_loras, lora_alpha=None)
        loaded_pipeline_config = current_config
    common_args_base = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(inference_steps),
        "tiled": tiled,
        "width": target_width,
        "height": target_height,
        "num_frames": int(num_frames),
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift
    }
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        return log_text
    if not os.path.exists(batch_output_folder):
        os.makedirs(batch_output_folder)
        log_text += f"[CMD] Created batch processing outputs folder: {batch_output_folder}\n"
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg", ".mp4"]]
    total_files = len(files)
    log_text += f"[CMD] Found {total_files} files in folder {folder_path}\n"
    seed_counter = 0
    for file in files:
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user.\n"
            if clear_cache_after_gen:
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cancel_batch_flag = False
            return log_text
        iter_start = time.time()
        base, ext = os.path.splitext(file)
        prompt_path = os.path.join(folder_path, base + ".txt")
        if not os.path.exists(prompt_path):
            log_text += f"[CMD] No prompt txt found for {file}, using user entered prompt: {default_prompt}\n"
            prompt_content = default_prompt
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            if prompt_content == "":
                log_text += f"[CMD] Prompt txt {base + '.txt'} is empty, using user entered prompt: {default_prompt}\n"
                prompt_content = default_prompt
            else:
                log_text += f"[CMD] Using user made prompt txt for {file}: {prompt_content}\n"
        final_prompt = process_random_prompt(prompt_content)
        output_filename = os.path.join(batch_output_folder, base + ".mp4")
        if skip_overwrite and os.path.exists(output_filename):
            log_text += f"[CMD] Output video {output_filename} already exists, skipping {file}.\n"
            continue
        if use_random_seed:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            try:
                base_seed = int(seed_input.strip()) if seed_input.strip() != "" else 0
                current_seed = base_seed + seed_counter
                seed_counter += 1
            except:
                current_seed = 0
        common_args = common_args_base.copy()
        common_args["prompt"] = final_prompt
        common_args["seed"] = current_seed
        if enable_teacache:
            common_args["tea_cache_l1_thresh"] = tea_cache_l1_thresh
            common_args["tea_cache_model_id"] = tea_cache_model_id
        else:
            common_args["tea_cache_l1_thresh"] = None
            common_args["tea_cache_model_id"] = ""
        log_text += f"[CMD] Processing {file} with prompt and seed {current_seed}\n"
        is_input_video = False
        image_obj = None
        if ext.lower() == ".mp4":
            is_input_video = True
            original_file_path = os.path.join(folder_path, file)
            image_obj = extract_last_frame(original_file_path)
            if image_obj is None:
                log_text += f"[CMD] Failed to extract frame from video {file}\n"
                continue
            log_text += f"[CMD] Extracted last frame from video {file} for processing.\n"
        else:
            try:
                image_path = os.path.join(folder_path, file)
                image_obj = Image.open(image_path)
                image_obj = ImageOps.exif_transpose(image_obj)
                image_obj = image_obj.convert("RGB")
            except Exception as e:
                log_text += f"[CMD] Failed to open image {file}: {str(e)}\n"
                continue
        if auto_crop:
            processed_image = auto_crop_image(image_obj, target_width, target_height)
        elif auto_scale:
            processed_image = auto_scale_image(image_obj, target_width, target_height)
        else:
            processed_image = image_obj
        if cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user before pipeline.\n"
            if clear_cache_after_gen:
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cancel_batch_flag = False
            return log_text
        video_data = loaded_pipeline(
            input_image=processed_image,
            **common_args,
            cancel_fn=lambda: cancel_batch_flag
        )
        if not video_data or cancel_batch_flag:
            log_text += "[CMD] Batch processing cancelled by user mid-run.\n"
            if clear_cache_after_gen:
                loaded_pipeline = None
                loaded_pipeline_config = {}
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cancel_batch_flag = False
            return log_text
        save_video(video_data, output_filename, fps=fps, quality=quality)
        log_text += f"[CMD] Saved batch generated video: {output_filename}\n"
        if auto_crop or auto_scale:
            preprocessed_folder = "auto_pre_processed_images"
            if not os.path.exists(preprocessed_folder):
                os.makedirs(preprocessed_folder)
            base_name = os.path.splitext(os.path.basename(output_filename))[0]
            preprocessed_image_filename = os.path.join(preprocessed_folder, f"{base_name}.png")
            processed_image.save(preprocessed_image_filename)
            log_text += f"[CMD] Saved auto processed image: {preprocessed_image_filename}\n"
        generation_duration = time.time() - iter_start
        if save_prompt:
            text_filename = os.path.splitext(output_filename)[0] + ".txt"
            generation_details = ""
            generation_details += f"Prompt: {final_prompt}\n"
            generation_details += f"Negative Prompt: {negative_prompt}\n"
            generation_details += f"Used Model: {model_choice_radio}\n"
            generation_details += f"Number of Inference Steps: {inference_steps}\n"
            generation_details += f"CFG Scale: {cfg_scale}\n"
            generation_details += f"Sigma Shift: {sigma_shift}\n"
            generation_details += f"Seed: {current_seed}\n"
            generation_details += f"Number of Frames: {num_frames}\n"
            generation_details += f"Denoising Strength: {denoising_strength}\n"
            if effective_loras:
                lora_details = ", ".join([f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras])
                generation_details += f"LoRA Models: {lora_details}\n"
            else:
                generation_details += "LoRA Model: None\n"
            generation_details += f"TeaCache Enabled: {enable_teacache}\n"
            if enable_teacache:
                generation_details += f"TeaCache L1 Threshold: {tea_cache_l1_thresh}\n"
                generation_details += f"TeaCache Model ID: {tea_cache_model_id}\n"
            generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
            generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
            generation_details += f"Final Resolution: {target_width}x{target_height}\n"
            generation_details += f"Generation Duration: {generation_duration:.2f} seconds / {(generation_duration/60):.2f} minutes\n"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(generation_details)
            log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
            # Apply Practical-RIFE for batch processed video if enabled and FPS <= 29.
            if pr_rife_enabled:
                cap = cv2.VideoCapture(output_filename)
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if source_fps <= 29:
                    multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                    improved_video = os.path.join(batch_output_folder, "improved_" + os.path.basename(output_filename))
                    model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                    cmd = (
                        f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
                        f'--model="{model_dir}" --multi={multiplier_val} '
                        f'--video="{output_filename}" --output="{improved_video}"'
                    )
                    subprocess.run(cmd, shell=True, check=True, env=os.environ)
                    log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                    output_filename = improved_video
                else:
                    log_text += f"[CMD] Skipping Practical-RIFE for batch generated video because its FPS ({source_fps:.2f}) is above 29.\n"
        if extend_factor > 1:
            if is_input_video:
                original_file_path = copy_to_outputs(original_file_path)
                segments = [original_file_path]
                cur_last_video = original_file_path
            else:
                segments = [output_filename]
                cur_last_video = output_filename
            if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
                ext_model_radio = "WAN 2.1 14B Image-to-Video 480P"
            elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
                ext_model_radio = "WAN 2.1 14B Image-to-Video 720P"
            else:
                ext_model_radio = model_choice_radio
            if ext_model_radio == "WAN 2.1 14B Image-to-Video 480P":
                ext_model_code = "14B_image_480p"
            elif ext_model_radio == "WAN 2.1 14B Image-to-Video 720P":
                ext_model_code = "14B_image_720p"
            else:
                ext_model_code = model_choice
            for ext_iter in range(1, int(extend_factor)):
                if ext_iter == 1 and is_input_video:
                    last_frame = extract_last_frame(original_file_path)
                else:
                    last_frame = extract_last_frame(cur_last_video)
                if last_frame is None:
                    log_text += "[CMD] Failed to extract last frame for extension.\n"
                    break
                used_folder = "used_last_frames"
                if not os.path.exists(used_folder):
                    os.makedirs(used_folder)
                base_name = os.path.splitext(os.path.basename(cur_last_video))[0]
                last_frame_filename = os.path.join(used_folder, f"{base_name}_lastframe.png")
                last_frame.save(last_frame_filename)
                log_text += f"[CMD] Saved used last frame: {last_frame_filename}\n"
                new_width, new_height = last_frame.size
                common_args_ext = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": int(inference_steps),
                    "seed": random.randint(0, 2**32 - 1) if use_random_seed else int(seed_input.strip() or 0) + ext_iter,
                    "tiled": tiled,
                    "width": new_width,
                    "height": new_height,
                    "num_frames": int(num_frames),
                    "cfg_scale": cfg_scale,
                    "sigma_shift": sigma_shift,
                }
                if enable_teacache:
                    common_args_ext["tea_cache_l1_thresh"] = tea_cache_l1_thresh
                    common_args_ext["tea_cache_model_id"] = tea_cache_model_id
                else:
                    common_args_ext["tea_cache_l1_thresh"] = None
                    common_args_ext["tea_cache_model_id"] = ""
                ext_config = {"model_choice": ext_model_code, "torch_dtype": torch_dtype, "num_persistent": vram_value}
                loaded_pipeline, loaded_pipeline_config = clear_pipeline_if_needed(loaded_pipeline, loaded_pipeline_config, ext_config)
                if loaded_pipeline is None:
                    loaded_pipeline = load_wan_pipeline(ext_model_code, torch_dtype, vram_value, lora_path=[], lora_alpha=None)
                    loaded_pipeline_config = ext_config
                log_text += f"[CMD] Processing extension for {file} extension iteration {ext_iter}\n"
                output_filename_ext = os.path.join(batch_output_folder, get_next_filename(".mp4"))
                video_data_ext = loaded_pipeline(
                    input_image=last_frame,
                    **common_args_ext,
                    cancel_fn=lambda: cancel_batch_flag
                )
                if not video_data_ext or cancel_batch_flag:
                    log_text += "[CMD] Extension generation cancelled by user mid-run.\n"
                    break
                save_video(video_data_ext, output_filename_ext, fps=fps, quality=quality)
                log_text += f"[CMD] Saved extension segment {ext_iter}: {output_filename_ext}\n"
                segments.append(output_filename_ext)
                cur_last_video = output_filename_ext
            if len(segments) > 1:
                merged_video = merge_videos(segments)
                log_text += f"[CMD] Merged extended video saved as: {merged_video}\n"
                output_filename = merged_video
            if pr_rife_enabled:
                cap = cv2.VideoCapture(output_filename)
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if source_fps <= 29:
                    multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                    improved_video = os.path.join(batch_output_folder, "improved_" + os.path.basename(output_filename))
                    model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                    cmd = (
                        f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
                        f'--model="{model_dir}" --multi={multiplier_val} '
                        f'--video="{output_filename}" --output="{improved_video}"'
                    )
                    subprocess.run(cmd, shell=True, check=True, env=os.environ)
                    log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                    output_filename = improved_video
                else:
                    log_text += f"[CMD] Skipping Practical-RIFE for batch generated video extension because its FPS ({source_fps:.2f}) is above 29.\n"
        if pr_rife_enabled:
            cap = cv2.VideoCapture(output_filename)
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if source_fps <= 29:
                multiplier_val = "2" if pr_rife_radio == "2x FPS" else "4"
                improved_video = os.path.join(batch_output_folder, "improved_" + os.path.basename(output_filename))
                model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
                cmd = (
                    f'"{sys.executable}" "Practical-RIFE/inference_video.py" '
                    f'--model="{model_dir}" --multi={multiplier_val} '
                    f'--video="{output_filename}" --output="{improved_video}"'
                )
                subprocess.run(cmd, shell=True, check=True, env=os.environ)
                log_text += f"[CMD] Applied Practical-RIFE with multiplier {multiplier_val}x. Improved video saved to {improved_video}\n"
                output_filename = improved_video
            else:
                log_text += f"[CMD] Skipping Practical-RIFE for batch generated video because its FPS ({source_fps:.2f}) is above 29.\n"
        generation_duration = time.time() - iter_start
        if save_prompt:
            text_filename = os.path.splitext(output_filename)[0] + ".txt"
            generation_details = ""
            generation_details += f"Prompt: {final_prompt}\n"
            generation_details += f"Negative Prompt: {negative_prompt}\n"
            generation_details += f"Used Model: {model_choice_radio}\n"
            generation_details += f"Number of Inference Steps: {inference_steps}\n"
            generation_details += f"CFG Scale: {cfg_scale}\n"
            generation_details += f"Sigma Shift: {sigma_shift}\n"
            generation_details += f"Seed: {current_seed}\n"
            generation_details += f"Number of Frames: {num_frames}\n"
            generation_details += f"Denoising Strength: {denoising_strength}\n"
            if effective_loras:
                lora_details = ", ".join([f"{os.path.basename(path)} (scale {alpha})" for path, alpha in effective_loras])
                generation_details += f"LoRA Models: {lora_details}\n"
            else:
                generation_details += "LoRA Model: None\n"
            generation_details += f"TeaCache Enabled: {enable_teacache}\n"
            if enable_teacache:
                generation_details += f"TeaCache L1 Threshold: {tea_cache_l1_thresh}\n"
                generation_details += f"TeaCache Model ID: {tea_cache_model_id}\n"
            generation_details += f"Precision: {'FP8' if torch_dtype == 'torch.float8_e4m3fn' else 'BF16'}\n"
            generation_details += f"Auto Crop: {'Enabled' if auto_crop else 'Disabled'}\n"
            generation_details += f"Final Resolution: {target_width}x{target_height}\n"
            generation_details += f"Generation Duration: {generation_duration:.2f} seconds / {(generation_duration/60):.2f} minutes\n"
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(generation_details)
            log_text += f"[CMD] Saved prompt and parameters: {text_filename}\n"
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
        except:
            continue
    next_number = max(current_numbers, default=0) + 1
    return os.path.join(outputs_dir, f"{next_number:05d}{extension}")

def open_outputs_folder():
    outputs_dir = os.path.abspath("outputs")
    if os.name == 'nt':
        os.startfile(outputs_dir)
    elif os.name == 'posix':
        subprocess.Popen(["xdg-open", outputs_dir])
    else:
        print("[CMD] Opening folder not supported on this OS.")
    return f"Opened {outputs_dir}"


model_manager = ModelManager(device="cpu")

def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent, lora_path=None, lora_alpha=None):
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16
    global model_manager
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
        model_manager.load_models([clip_path], torch_dtype=torch.float32)
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
        model_manager.load_models([clip_path], torch_dtype=torch.float32)
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
                t5_path,
                vae_path,
            ],
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError("Invalid model choice")
    if lora_path is not None:
        if isinstance(lora_path, list):
            for path, alpha in lora_path:
                print(f"[CMD] Loading LoRA from {path} with alpha {alpha}")
                model_manager.load_lora(path, lora_alpha=alpha)
        else:
            print(f"[CMD] Loading LoRA from {lora_path} with alpha {lora_alpha}")
            model_manager.load_lora(lora_path, lora_alpha=lora_alpha)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    try:
        num_persistent_val = int(num_persistent)
    except:
        print("[CMD] Warning: could not parse num_persistent value, defaulting to 6000000000")
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

def apply_fast_preset():
    return 20, True, 0.15, 10

# ------------------------- Main application with Gradio -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    args = parser.parse_args()
    loaded_pipeline = None
    loaded_pipeline_config = {}
    cancel_flag = False
    cancel_batch_flag = False
    prompt_expander = None
    with gr.Blocks() as demo:
        gr.Markdown("SECourses Wan 2.1 I2V - V2V - T2V Advanced Gradio APP V46 | Tutorial : https://youtu.be/hnAhveNy-8s | Source : https://www.patreon.com/posts/123105403")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    cancel_button = gr.Button("Cancel")
                    fast_preset_button = gr.Button("Apply Fast Preset", variant="huggingface")
                    enhance_button = gr.Button("Prompt Enhance", variant="primary")
                prompt_box = gr.Textbox(label="Prompt (A <random: green , yellow , etc > car) will take random word with trim like : A yellow car", placeholder="Describe the video you want to generate", lines=5)                
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
                    extend_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("extend_factor", 1), label="Extend Video Factor (1× = No Extension)")
                    extension_info_button = gr.Button("Extension Feature Info")
                with gr.Row():
                    extension_info_output = gr.Markdown("")
                with gr.Row():
                    num_generations = gr.Number(label="Number of Generations (e.g. Generate 3 videos)", value=config_loaded.get("num_generations", 1), precision=0)
                    width_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("width", 832), label="Width")
                    height_slider = gr.Slider(minimum=320, maximum=1536, step=16, value=config_loaded.get("height", 480), label="Height")
                with gr.Row():
                    clear_cache_checkbox = gr.Checkbox(label="Clear model from RAM and VRAM after generation", value=config_loaded.get("clear_cache_after_gen", DEFAULT_CLEAR_CACHE))
                    
                    with gr.Column():
                        auto_crop_checkbox = gr.Checkbox(label="Auto Crop", value=config_loaded.get("auto_crop", True))
                        auto_scale_checkbox = gr.Checkbox(label="Auto Scale", value=config_loaded.get("auto_scale", False))
                    
                    tiled_checkbox = gr.Checkbox(label="Tiled VAE Decode (Disable for 1.3B model for 12GB or more GPUs)", value=config_loaded.get("tiled", True))
                    inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=config_loaded.get("inference_steps", 50), label="Inference Steps")
                with gr.Row():
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=config_loaded.get("quality", 5), label="Quality")
                    fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=config_loaded.get("fps", 16), label="FPS (for saving video - you can save as 8 FPS and 4x RIFE to get 2x duration)")
                    num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=config_loaded.get("num_frames", 81), label="Number of Frames (Always 4x+1 e.g. 17 frames = 1 second). More frames uses more VRAM and slower")
                gr.Markdown("### Increase Video FPS with Practical-RIFE")
                with gr.Row():
                    pr_rife_checkbox = gr.Checkbox(label="Apply Practical-RIFE", value=config_loaded.get("pr_rife", True))
                    pr_rife_radio = gr.Radio(choices=["2x FPS", "4x FPS"], label="FPS Multiplier", value=config_loaded.get("pr_rife_multiplier", "2x FPS"))
                    cfg_scale_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("cfg_scale", 6.0), label="CFG Scale")
                    sigma_shift_slider = gr.Slider(minimum=3, maximum=12, step=0.1, value=config_loaded.get("sigma_shift", 6.0), label="Sigma Shift")
                gr.Markdown("### GPU Settings - If you get out of VRAM error or if it uses shared VRAM, reduce this number, FP8 may generate color broken at the moment in I2V")
                with gr.Row():
                    num_persistent_text = gr.Textbox(label="Number of Persistent Parameters In Dit (VRAM)", value=config_loaded.get("num_persistent", "12000000000"))
                    torch_dtype_radio = gr.Radio(
                        choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                        label="torch.float8_e4m3fn is FP8 and reduces VRAM and RAM usage a lot with little quality loss. torch.bfloat16 is BF16 (max quality)",
                        value=config_loaded.get("torch_dtype", "torch.bfloat16")
                    )
                gr.Markdown("### TeaCache Settings - Significantly speeds up generation as it progress - Too big value reduces quality and causes distortions")
                with gr.Row():
                    enable_teacache_checkbox = gr.Checkbox(label="Enable TeaCache (0.05 Threshold for 1.3b model and 0.15 for 14b models recommended)", value=config_loaded.get("enable_teacache", False))
                with gr.Row():
                    tea_cache_l1_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=config_loaded.get("tea_cache_l1_thresh", 0.15), label="Tea Cache L1 Threshold")
                    tea_cache_model_id_textbox = gr.Textbox(label="Tea Cache Model ID", value=config_loaded.get("tea_cache_model_id", "Wan2.1-T2V-1.3B"), placeholder="Enter Tea Cache Model ID")
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        label="LoRA Model (Place .safetensors files in 'LoRAs' folder)",
                        choices=get_lora_choices(),
                        value=config_loaded.get("lora_model", "None")
                    )
                    lora_alpha_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha", 1.0), label="LoRA Scale")
                    refresh_lora_button = gr.Button("Refresh LoRAs")
                with gr.Row():
                    show_more_lora_button = gr.Button("Show More LoRAs")
                    more_lora_state = gr.State(False)
                with gr.Column(visible=False) as more_lora_container:
                    lora_dropdown_2 = gr.Dropdown(label="LoRA Model 2", choices=get_lora_choices(), value=config_loaded.get("lora_model_2", "None"))
                    lora_alpha_slider_2 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_2", 1.0), label="LoRA Scale 2")
                    lora_dropdown_3 = gr.Dropdown(label="LoRA Model 3", choices=get_lora_choices(), value=config_loaded.get("lora_model_3", "None"))
                    lora_alpha_slider_3 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_3", 1.0), label="LoRA Scale 3")
                    lora_dropdown_4 = gr.Dropdown(label="LoRA Model 4", choices=get_lora_choices(), value=config_loaded.get("lora_model_4", "None"))
                    lora_alpha_slider_4 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=config_loaded.get("lora_alpha_4", 1.0), label="LoRA Scale 4")
                show_more_lora_button.click(fn=toggle_lora_visibility, inputs=[more_lora_state], outputs=[more_lora_container, more_lora_state, show_more_lora_button])
                with gr.Row():
                    gr.Markdown("Target language for prompt enhance:") 
                    tar_lang = gr.Radio(choices=["CH", "EN"], container=False, label="Target language for prompt enhance", value=config_loaded.get("tar_lang", "EN"))
                negative_prompt = gr.Textbox(label="Negative Prompt", value=config_loaded.get("negative_prompt", "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"), placeholder="Enter negative prompt", lines=2)
                with gr.Row():
                    save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=config_loaded.get("save_prompt", True))
                    multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=config_loaded.get("multiline", False))                    
                    use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=config_loaded.get("use_random_seed", True))
                    seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value=config_loaded.get("seed", ""))
                gr.Markdown("### Use Left Panel to Upload Image to Video, Right Panel to Upload Video to Video (1.3b Model) or Extent Existing Video (480p and 720p I2V models)")
                with gr.Row():
                    denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=config_loaded.get("denoising_strength", 0.7),
                                             label="Denoising Strength (only for video-to-video)")
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                    video_input = gr.Video(label="Input Video (for Video-to-Video, only for 1.3B) or Extending Existing Video (Uses Last Frame, for Image-to-Video models)", format="mp4", height=512)

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
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        vram_preset_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        torch_dtype_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio, torch_dtype_radio],
            outputs=[aspect_ratio_radio, width_slider, height_slider, num_persistent_text]
        )
        aspect_ratio_radio.change(
            fn=update_width_height,
            inputs=[aspect_ratio_radio, model_choice_radio],
            outputs=[width_slider, height_slider]
        )
        model_choice_radio.change(
            fn=update_tea_cache_model_id,
            inputs=[model_choice_radio],
            outputs=[tea_cache_model_id_textbox]
        )
        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[
                prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider,
                num_generations, save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider,
                model_choice_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio,
                num_frames_slider,
                aspect_ratio_radio, width_slider, height_slider, auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox,
                inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                extend_slider
            ],
            outputs=[video_output, status_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output)
        fast_preset_button.click(fn=apply_fast_preset, inputs=[], outputs=[inference_steps_slider, enable_teacache_checkbox , tea_cache_l1_thresh_slider, sigma_shift_slider])
        open_outputs_button.click(fn=open_outputs_folder, outputs=status_output)
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=[
                prompt_box,
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
                auto_scale_checkbox,
                tiled_checkbox,
                cfg_scale_slider,
                sigma_shift_slider,
                save_prompt_batch_checkbox,
                pr_rife_checkbox, 
                pr_rife_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                enable_teacache_checkbox, 
                tea_cache_l1_thresh_slider, 
                tea_cache_model_id_textbox,
                clear_cache_checkbox,
                extend_slider
            ],
            outputs=batch_status_output
        )
        cancel_batch_process_button.click(fn=cancel_batch_process, outputs=batch_status_output)
        load_config_button.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[
                config_status, 
                model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                negative_prompt, save_prompt_checkbox, multiline_checkbox, num_generations, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                extend_slider
            ]
        )
        save_config_button.click(
            fn=save_config,
            inputs=[
                config_name_textbox, model_choice_radio, vram_preset_radio, aspect_ratio_radio, width_slider, height_slider,
                auto_crop_checkbox, auto_scale_checkbox, tiled_checkbox, inference_steps_slider, pr_rife_checkbox, pr_rife_radio, cfg_scale_slider, sigma_shift_slider,
                num_persistent_text, torch_dtype_radio,
                lora_dropdown, lora_alpha_slider,
                lora_dropdown_2, lora_alpha_slider_2,
                lora_dropdown_3, lora_alpha_slider_3,
                lora_dropdown_4, lora_alpha_slider_4,
                clear_cache_checkbox,
                negative_prompt, save_prompt_checkbox, multiline_checkbox, num_generations, use_random_seed_checkbox, seed_input,
                quality_slider, fps_slider, num_frames_slider, denoising_slider, tar_lang,
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, save_prompt_batch_checkbox,
                enable_teacache_checkbox, tea_cache_l1_thresh_slider, tea_cache_model_id_textbox,
                extend_slider
            ],
            outputs=[config_status, config_dropdown]
        )
        image_input.change(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        auto_scale_checkbox.change(
            fn=update_target_dimensions,
            inputs=[image_input, auto_scale_checkbox, width_slider, height_slider],
            outputs=[width_slider, height_slider]
        )
        extension_info_button.click(fn=show_extension_info, inputs=[], outputs=[extension_info_output])
        demo.launch(share=args.share, inbrowser=True)