import os
import argparse
import shutil
import json
from pathlib import Path

# 导入App.py中的批处理函数
from App import batch_process_videos, get_default_config

def setup_batch_folder(images_dir, prompts_file, batch_folder="batch_inputs"):
    """
    设置批处理文件夹，复制图片并创建提示文本文件
    
    Args:
        images_dir: 图像目录
        prompts_file: 包含提示的JSON文件 {filename: prompt}
        batch_folder: 批处理输入文件夹
    """
    # 确保批处理文件夹存在
    os.makedirs(batch_folder, exist_ok=True)
    
    # 加载提示
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    # 处理每个图像和对应的提示
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # 源文件路径
            src_path = os.path.join(images_dir, img_file)
            # 目标文件路径
            dst_path = os.path.join(batch_folder, img_file)
            
            # 复制图像文件
            shutil.copy2(src_path, dst_path)
            
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(img_file)[0]
            
            # 查找对应的提示
            prompt = prompts.get(img_file) or prompts.get(base_name, "A high quality video")
            
            # 创建提示文本文件
            with open(os.path.join(batch_folder, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                f.write(prompt)
    
    print(f"已准备 {len(os.listdir(images_dir))} 个文件到批处理文件夹 {batch_folder}")
    return batch_folder

def main():
    parser = argparse.ArgumentParser(description="批量处理图像和提示生成视频")
    parser.add_argument("--images", type=str, required=True, help="包含图像的目录")
    parser.add_argument("--prompts", type=str, required=True, help="包含提示的JSON文件")
    parser.add_argument("--config", type=str, help="配置文件路径，默认使用内置配置")
    parser.add_argument("--batch_folder", type=str, default="batch_inputs", help="批处理输入文件夹")
    parser.add_argument("--output_folder", type=str, default="batch_outputs", help="批处理输出文件夹")
    parser.add_argument("--negative_prompt", type=str, help="负面提示")
    parser.add_argument("--model", type=str, choices=[
        "WAN 2.1 1.3B (Text/Video-to-Video)",
        "WAN 2.1 14B Text-to-Video",
        "WAN 2.1 14B Image-to-Video 720P",
        "WAN 2.1 14B Image-to-Video 480P"
    ], default="WAN 2.1 14B Image-to-Video 720P", help="模型选择")
    
    args = parser.parse_args()
    
    # 设置批处理文件夹
    batch_folder = setup_batch_folder(args.images, args.prompts, args.batch_folder)
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("找不到config文件")
    
    # 更新批处理相关配置
    config["batch_folder"] = batch_folder
    config["batch_output_folder"] = args.output_folder
    
    # 如果提供了负面提示，则更新配置
    if args.negative_prompt:
        config["negative_prompt"] = args.negative_prompt
    
    # 调用批处理函数
    print("开始批量处理...")
    log = batch_process_videos(
        default_prompt="A high quality video",  # 默认提示，将被各个文件的提示覆盖
        folder_path=config["batch_folder"],
        batch_output_folder=config["batch_output_folder"],
        skip_overwrite=config["skip_overwrite"],
        tar_lang=config["tar_lang"],
        negative_prompt=config["negative_prompt"],
        denoising_strength=config["denoising_strength"],
        use_random_seed=config["use_random_seed"],
        seed_input=config["seed"],
        quality=config["quality"],
        fps=config["fps"],
        model_choice_radio=config["model_choice"],
        vram_preset=config["vram_preset"],
        num_persistent_input=config["num_persistent"],
        torch_dtype=config["torch_dtype"],
        num_frames=config["num_frames"],
        inference_steps=config["inference_steps"],
        aspect_ratio=config["aspect_ratio"],
        width=config["width"],
        height=config["height"],
        auto_crop=config["auto_crop"],
        auto_scale=config["auto_scale"],
        tiled=config["tiled"],
        cfg_scale=config["cfg_scale"],
        sigma_shift=config["sigma_shift"],
        save_prompt=config["save_prompt"],
        pr_rife_enabled=config["pr_rife"],
        pr_rife_radio=config["pr_rife_multiplier"],
        lora_model=config["lora_model"],
        lora_alpha=config["lora_alpha"],
        lora_model_2=config["lora_model_2"],
        lora_alpha_2=config["lora_alpha_2"],
        lora_model_3=config["lora_model_3"],
        lora_alpha_3=config["lora_alpha_3"],
        lora_model_4=config["lora_model_4"],
        lora_alpha_4=config["lora_alpha_4"],
        enable_teacache=config["enable_teacache"],
        tea_cache_l1_thresh=config["tea_cache_l1_thresh"],
        tea_cache_model_id=config["tea_cache_model_id"],
        clear_cache_after_gen=config["clear_cache_after_gen"],
        extend_factor=config["extend_factor"],
        num_generations=config["num_generations"]
    )
    
    print("批处理完成！")
    print(f"输出视频在: {os.path.abspath(args.output_folder)}")
    
    # 可选：将处理日志保存到文件
    with open("batch_process_log.txt", "w", encoding="utf-8") as f:
        f.write(log)

if __name__ == "__main__":
    main()