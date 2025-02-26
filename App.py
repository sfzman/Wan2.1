import os
import sys
import subprocess
import random
import argparse
import time

import torch
import gradio as gr
from PIL import Image

# Import WAN prompt extend components
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

# Import DiffSynth and related video generation components
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download

# Global variables for prompt expansion and pipeline persistence
prompt_expander = None
loaded_pipeline = None
loaded_pipeline_config = {}  # To check if pipeline already loaded with the same settings
cancel_flag = False  # Flag to cancel long generation loops
cancel_batch_flag = False  # Flag to cancel batch processing loops


#############################################
# Update GPU VRAM preset and resolution options
#############################################
def update_vram_and_resolution(model_choice, preset):
    """
    Maps GPU VRAM preset names to a num_persistent_param_in_dit value based on the selected model.
    
    For 1.3B model:
      - "4GB", "6GB", "8GB", ... => specific values and resolution choices.
    
    For 14B Text-to-Video model:
      - Most lower VRAM presets map to "0" until "48GB" and "80GB" which map to large numbers.
      
    For 14B Image-to-Video model (new separation):
      - Uses a new mapping with different values and sets a different default resolution.
    """
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
            "48GB": "12000000000",
            "80GB": "12000000000"
        }
        resolution_choices = ["832x480", "480x832"]
        default_resolution = "480x832"
    elif model_choice == "WAN 2.1 14B Text-to-Video":
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "3000000000",
            "48GB": "22000000000",
            "80GB": "70000000000"
        }
        resolution_choices = ["1280x720", "720x1280"]
        default_resolution = "720x1280"
    elif model_choice == "WAN 2.1 14B Image-to-Video":  # New separation for image-to-video models
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = ["1280x720", "720x1280"]
        default_resolution = "1280x720"  # New default resolution for 14B Image-to-Video
    else:
        mapping = {
            "4GB": "0",
            "6GB": "0",
            "8GB": "0",
            "10GB": "0",
            "12GB": "0",
            "16GB": "0",
            "24GB": "0",
            "48GB": "12000000000",
            "80GB": "70000000000"
        }
        resolution_choices = ["1280x720", "720x1280"]
        default_resolution = "720x1280"
    return mapping.get(preset, "12000000000"), resolution_choices, default_resolution


def update_model_settings(model_choice, current_vram_preset):
    """
    When the model selection changes, update the resolution radio options
    and the num_persistent text field.
    """
    num_persistent_val, resolution_choices, default_resolution = update_vram_and_resolution(model_choice, current_vram_preset)
    return gr.update(choices=resolution_choices, value=default_resolution), num_persistent_val


def update_vram_on_change(preset, model_choice):
    """
    When the VRAM preset changes, update the num_persistent text field based on the current model.
    """
    num_persistent_val, _, _ = update_vram_and_resolution(model_choice, preset)
    return num_persistent_val


#############################################
# Prompt and Pipeline Helper Functions
#############################################
def prompt_enc(prompt, tar_lang):
    """
    Enhances the prompt using the prompt expander model.
    Before enhancing, the loaded WAN pipeline (if any) is cleared to free VRAM.
    After enhancement, the prompt expander is also cleared so that the WAN pipeline will reload on video generation.
    """
    global prompt_expander, loaded_pipeline, loaded_pipeline_config, args

    # Clear the WAN pipeline before running prompt enhancement (free up VRAM)
    if loaded_pipeline is not None:
        print("[CMD] Clearing loaded WAN pipeline before prompt enhancement.")
        loaded_pipeline = None
        loaded_pipeline_config = {}

    # Load the prompt expander if needed
    if prompt_expander is None:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl=False)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl=False, device=0)
        else:
            raise NotImplementedError(
                f"Unsupported prompt_extend_method: {args.prompt_extend_method}"
            )
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    result = prompt if not prompt_output.status else prompt_output.prompt

    # Clear the prompt expander after using it to ensure reloading later
    prompt_expander = None

    return result


def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent):
    """
    Loads the appropriate WAN pipeline based on:
      - model_choice: one of "1.3B", "14B_text", or "14B_image"
      - torch_dtype_str: either "torch.float8_e4m3fn" or "torch.bfloat16"
      - num_persistent: VRAM related parameter (can be an integer or None)
    """
    print(f"[CMD] Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16

    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        # 1.3B text-to-video and video-to-video pipeline
        model_manager.load_models(
            [
                "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device)
    elif model_choice == "14B_text":
        # 14B text-to-video pipeline
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
                ],
                "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device)
    elif model_choice == "14B_image":
        # 14B image-to-video pipeline (same as before, but now with separate configuration values)
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device)
    else:
        raise ValueError("Invalid model choice")
    
    # Convert the num_persistent value: if "None" (string) then use None; otherwise, cast to int.
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


def get_next_filename(extension):
    """
    Returns the next available file path under the "outputs" folder.
    The filename is in the format 00001.ext, 00002.ext, etc.
    """
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


#############################################
# Main Generation Function
#############################################
def generate_videos(
    prompt, tar_lang, negative_prompt, input_image, input_video, denoising_strength, num_generations,
    save_prompt, multi_line, use_random_seed, seed_input, quality, fps,
    model_choice_radio, resolution_radio, vram_preset, num_persistent_input, torch_dtype, num_frames, inference_steps
):
    """
    Single generation function that:
      - Loads the pipeline on the first run (or if configuration changes)
      - Handles multi-line prompts if enabled
      - Supports Text-to-Video, Video-to-Video (for 1.3B) and Image-to-Video (for 14B)
      - Saves each generated video to the outputs folder and optionally writes out prompts to a .txt file
      - Returns the last generated video path, a status log, and the last used seed.
      - Now includes a user-configurable "num_inference_steps" value.
    """
    global loaded_pipeline, loaded_pipeline_config, cancel_flag
    cancel_flag = False  # reset cancellation flag at start
    log_text = ""
    last_used_seed = None
    last_video_path = ""
    start_time = time.time()  # start timer

    if model_choice_radio == "WAN 2.1 1.3B (Text/Video-to-Video)":
        model_choice = "1.3B"
    elif model_choice_radio == "WAN 2.1 14B Text-to-Video":
        model_choice = "14B_text"
    elif model_choice_radio == "WAN 2.1 14B Image-to-Video":
        model_choice = "14B_image"
    else:
        return "", "Invalid model choice.", ""
    
    # Determine resolution
    if model_choice in ["14B_text", "14B_image"]:
        if resolution_radio == "1280x720":
            width, height = 1280, 720
        else:
            width, height = 720, 1280
    else:  # 1.3B
        if resolution_radio == "832x480":
            width, height = 832, 480
        else:
            width, height = 480, 832

    # Use the VRAM preset text value
    vram_value = num_persistent_input

    # Load the pipeline if not already loaded or if configuration has changed.
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value)
        loaded_pipeline_config = current_config

    # Prepare prompts list
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
                duration = time.time() - start_time
                log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
                log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
                return "", log_text, str(last_used_seed or "")
            iteration += 1
            log_text += f"[CMD] Generating video {iteration} of {total_iterations} with prompt: {p}\n"
            print(f"[CMD] Generating video {iteration}/{total_iterations} with prompt: {p}")

            # Optionally enhance prompt
            enhanced_prompt = p

            # Determine seed
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                try:
                    current_seed = int(seed_input) if seed_input.strip() != "" else 0
                except:
                    current_seed = 0
            last_used_seed = current_seed
            print(f"width: {width} : height {height}")

            # Build common generation parameters
            common_args = {
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled": True,
                "width": width,
                "height": height,
                "num_frames": int(num_frames),
            }

            # Choose pipeline call based on model and inputs.
            if model_choice == "1.3B":
                if input_video is not None:
                    input_video_path = input_video if isinstance(input_video, str) else input_video.name
                    video_obj = VideoData(input_video_path, height=height, width=width)
                    video_data = loaded_pipeline(input_video=video_obj, denoising_strength=denoising_strength, **common_args)
                else:
                    video_data = loaded_pipeline(**common_args)
            elif model_choice == "14B_text":
                video_data = loaded_pipeline(**common_args)
            elif model_choice == "14B_image":
                if input_image is None:
                    err_msg = "[CMD] Error: Image model selected but no image provided."
                    print(err_msg)
                    return "", err_msg, str(last_used_seed or "")
                video_data = loaded_pipeline(input_image=input_image, **common_args)
            else:
                err_msg = "[CMD] Invalid combination of inputs."
                print(err_msg)
                return "", err_msg, str(last_used_seed or "")

            # Save the generated video.
            video_filename = get_next_filename(".mp4")
            save_video(video_data, video_filename, fps=fps, quality=quality)
            log_text += f"[CMD] Saved video: {video_filename}\n"
            print(f"[CMD] Saved video: {video_filename}")

            # Optionally, save prompt to a text file.
            if save_prompt:
                text_filename = os.path.splitext(video_filename)[0] + ".txt"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(enhanced_prompt)
                log_text += f"[CMD] Saved prompt: {text_filename}\n"
                print(f"[CMD] Saved prompt: {text_filename}")

            last_video_path = video_filename

    duration = time.time() - start_time
    log_text += f"\n[CMD] Used VRAM Setting: {vram_value}\n"
    log_text += f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}\n"
    print(f"[CMD] Generation complete. Duration: {duration:.2f} seconds. Last used seed: {last_used_seed}")
    return last_video_path, log_text, str(last_used_seed or "")


def cancel_generation():
    """
    Sets the global cancel flag so that generation loops can end early.
    """
    global cancel_flag
    cancel_flag = True
    print("[CMD] Cancel button pressed.")
    return "Cancelling generation..."


#############################################
# Batch Image-to-Video Processing Functionality
#############################################
def batch_process_videos(
    folder_path, batch_output_folder, skip_overwrite, tar_lang, negative_prompt, denoising_strength,
    use_random_seed, seed_input, quality, fps, model_choice_radio, resolution_radio,
    vram_preset, num_persistent_input, torch_dtype, num_frames, inference_steps
):
    """
    Processes a folder of images for image-to-video generation in batch.
    
    For each image file (jpg/png/jpeg) in the input folder, it attempts to find a corresponding
    .txt file (with the same base name) to obtain the prompt. If no prompt file exists or it's empty,
    the image is skipped.
    
    The generated video is saved in the user-specified "batch processing outputs folder" with the same
    base name as the input image.
    
    If "skip overwrite" is enabled and an output video already exists in the batch outputs folder,
    the image is skipped.
    
    NOTE: Batch processing is currently only supported for the "WAN 2.1 14B Image-to-Video" model.
    """
    global loaded_pipeline, loaded_pipeline_config, cancel_batch_flag
    cancel_batch_flag = False  # reset cancellation flag for batch process
    log_text = ""
    
    # Ensure batch processing is run only with 14B Image-to-Video model.
    if model_choice_radio != "WAN 2.1 14B Image-to-Video":
        log_text += "[CMD] Batch processing currently only supports the WAN 2.1 14B Image-to-Video model.\n"
        return log_text

    # Determine resolution for 14B models.
    if resolution_radio == "1280x720":
        width, height = 1280, 720
    else:
        width, height = 720, 1280

    vram_value = num_persistent_input

    # For batch processing, use the "14B_image" pipeline.
    model_choice = "14B_image"
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_value,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_value)
        loaded_pipeline_config = current_config

    common_args_base = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(inference_steps),
        "tiled": True,
        "width": width,
        "height": height,
        "num_frames": int(num_frames),
    }
    
    if not os.path.isdir(folder_path):
        log_text += f"[CMD] Provided folder path does not exist: {folder_path}\n"
        return log_text

    # Ensure the batch processing outputs folder exists (or create it)
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

        base, ext = os.path.splitext(image_file)
        prompt_path = os.path.join(folder_path, base + ".txt")
        if not os.path.exists(prompt_path):
            log_text += f"[CMD] No corresponding prompt file for {image_file}, skipping.\n"
            print(f"[CMD] No corresponding prompt file for {image_file}, skipping.")
            continue
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read().strip()
        if prompt_content == "":
            log_text += f"[CMD] Prompt file {base + '.txt'} is empty, skipping {image_file}.\n"
            print(f"[CMD] Prompt file {base + '.txt'} is empty, skipping {image_file}.")
            continue

        # Determine the output filename based on the image base name.
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
            except:
                current_seed = 0

        common_args = common_args_base.copy()
        common_args["prompt"] = prompt_content
        common_args["seed"] = current_seed

        log_text += f"[CMD] Processing {image_file} with prompt from {base + '.txt'} and seed {current_seed}\n"
        print(f"[CMD] Processing {image_file} with prompt from {base + '.txt'} and seed {current_seed}")
        
        try:
            image_path = os.path.join(folder_path, image_file)
            image_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            log_text += f"[CMD] Failed to open image {image_file}: {str(e)}\n"
            print(f"[CMD] Failed to open image {image_file}: {str(e)}")
            continue

        video_data = loaded_pipeline(input_image=image_obj, **common_args)
        save_video(video_data, output_filename, fps=fps, quality=quality)
        log_text += f"[CMD] Saved batch generated video: {output_filename}\n"
        print(f"[CMD] Saved batch generated video: {output_filename}")

    return log_text


def cancel_batch_process():
    """
    Sets the global cancel flag so that batch processing can end early.
    """
    global cancel_batch_flag
    cancel_batch_flag = True
    print("[CMD] Batch process cancel button pressed.")
    return "Cancelling batch process..."


#############################################
# Gradio Interface
#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Prompt Enhance arguments
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"],
                        help="The prompt extend method to use.")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
    # Gradio share option
    parser.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("SECourses Wan 2.1 I2V - V2V - T2V Advanced Gradio APP : https://www.patreon.com/posts/123105403")
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
                            "WAN 2.1 14B Image-to-Video"
                        ],
                        label="Model Choice",
                        value="WAN 2.1 1.3B (Text/Video-to-Video)"
                    )
                    resolution_radio = gr.Radio(
                        choices=["832x480", "480x832"],
                        label="Resolution (for 1.3B model)",
                        value="480x832"
                    )
                # Generate and Cancel buttons.
                gr.Markdown("### GPU Settings")
                with gr.Row():
                    vram_preset_radio = gr.Radio(
                        choices=["4GB", "6GB", "8GB", "10GB", "12GB", "16GB", "24GB", "48GB", "80GB"],
                        label="GPU VRAM Preset",
                        value="48GB"
                    )
                    num_persistent_text = gr.Textbox(label="Number of Persistent Parameters In Dit (VRAM)", value="12000000000")
                    torch_dtype_radio = gr.Radio(
                        choices=["torch.float8_e4m3fn", "torch.bfloat16"],
                        label="Torch DType (float8_e4m3fn not working yet)",
                        value="torch.bfloat16"
                    )                
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    cancel_button = gr.Button("Cancel")
                # Prompt input and enhancement.
                prompt_box = gr.Textbox(label="Prompt", placeholder="Describe the video you want to generate", lines=5)
                with gr.Row():
                    tar_lang = gr.Radio(choices=["CH", "EN"], label="Target language for prompt enhance", value="EN")
                    enhance_button = gr.Button("Prompt Enhance")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt", lines=2)
                # Save prompt and Multi-line prompt options.
                with gr.Row():
                    save_prompt_checkbox = gr.Checkbox(label="Save prompt to file", value=True)
                    multiline_checkbox = gr.Checkbox(label="Multi-line prompt (each line is separate)", value=False)
                num_generations = gr.Number(label="Number of Generations", value=1, precision=0)
                # Random seed options.
                with gr.Row():
                    use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True)
                    seed_input = gr.Textbox(label="Seed (if not using random)", placeholder="Enter seed", value="")
                # Quality, FPS, Number of Frames.
                with gr.Row():
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Quality")
                    fps_slider = gr.Slider(minimum=8, maximum=30, step=1, value=16, label="FPS (for saving video)")
                    num_frames_slider = gr.Slider(minimum=1, maximum=300, step=1, value=81, label="Number of Frames")
                # Inference Steps slider (new)
                inference_steps_slider = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Inference Steps")
                # Image and Video inputs.
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Input Image (for image-to-video)", height=512)
                    video_input = gr.Video(label="Input Video (for video-to-video, only for 1.3B)", format="mp4", height=512)
                denoising_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                                             label="Denoising Strength (only for video-to-video)")
                # GPU Settings.

            with gr.Column(scale=3):
                # Output Column: Generated video, then batch processing section, then logs.
                video_output = gr.Video(label="Generated Video", height=720)
                
                gr.Markdown("### Batch Image-to-Video Processing")
                batch_folder_input = gr.Textbox(label="Input Folder for Batch Processing", placeholder="Enter input folder path", value="batch_inputs")
                batch_output_folder_input = gr.Textbox(label="Batch Processing Outputs Folder", placeholder="Enter batch outputs folder path", value="batch_outputs")
                skip_overwrite_checkbox = gr.Checkbox(label="Skip Overwrite if Output Exists", value=True)
                with gr.Row():
                    batch_process_button = gr.Button("Batch Process")
                    cancel_batch_process_button = gr.Button("Cancel Batch Process")
                batch_status_output = gr.Textbox(label="Batch Process Status Log", lines=10)
                
                status_output = gr.Textbox(label="Status Log", lines=20)
                last_seed_output = gr.Textbox(label="Last Used Seed", interactive=False)
                open_outputs_button = gr.Button("Open Outputs Folder")

        # Button events and interactions.
        enhance_button.click(fn=prompt_enc, inputs=[prompt_box, tar_lang], outputs=prompt_box)
        generate_button.click(
            fn=generate_videos,
            inputs=[
                prompt_box, tar_lang, negative_prompt, image_input, video_input, denoising_slider, num_generations,
                save_prompt_checkbox, multiline_checkbox, use_random_seed_checkbox, seed_input, quality_slider, fps_slider,
                model_choice_radio, resolution_radio, vram_preset_radio, num_persistent_text, torch_dtype_radio, num_frames_slider, inference_steps_slider
            ],
            outputs=[video_output, status_output, last_seed_output]
        )
        cancel_button.click(fn=cancel_generation, outputs=status_output)
        open_outputs_button.click(fn=open_outputs_folder, outputs=status_output)
        batch_process_button.click(
            fn=batch_process_videos,
            inputs=[
                batch_folder_input, batch_output_folder_input, skip_overwrite_checkbox, tar_lang, negative_prompt, denoising_slider,
                use_random_seed_checkbox, seed_input, quality_slider, fps_slider, model_choice_radio, resolution_radio,
                vram_preset_radio, num_persistent_text, torch_dtype_radio, num_frames_slider, inference_steps_slider
            ],
            outputs=batch_status_output
        )
        cancel_batch_process_button.click(fn=cancel_batch_process, outputs=batch_status_output)
        model_choice_radio.change(
            fn=update_model_settings,
            inputs=[model_choice_radio, vram_preset_radio],
            outputs=[resolution_radio, num_persistent_text]
        )
        vram_preset_radio.change(
            fn=update_vram_on_change,
            inputs=[vram_preset_radio, model_choice_radio],
            outputs=num_persistent_text
        )

        demo.launch(share=args.share, inbrowser=True)
