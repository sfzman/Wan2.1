# WAN 2.1 批量图像到视频处理脚本

此脚本用于批量处理图像到视频的转换，基于WAN 2.1模型和Gradio界面的后端功能。

## 使用方法

### 1. 准备数据

- 将所有图像文件放在一个文件夹中（例如 `images/`）
- 创建一个JSON文件，映射图像文件名到提示文本，例如：

```json
{
    "image1.jpg": "一只优雅的猫咪在阳光下散步",
    "image2.jpg": "未来城市的街道，霓虹灯闪烁",
    "sunset_view.png": "日落时分的海滩，海浪轻轻拍打岸边"
}
```

注意：JSON中的键可以是完整文件名（带扩展名）或者不带扩展名的基本名称。

### 2. 运行脚本

基本用法：

```bash
python batch_process.py --images images/ --prompts example_prompts.json --config example_config.json
```

### 3. 参数说明

- `--images`：包含图像的目录路径（必填）
- `--prompts`：包含提示的JSON文件路径（必填）
- `--config`：配置文件路径（必填，使用内置默认配置）
- `--batch_folder`：批处理输入文件夹（默认：batch_inputs）
- `--output_folder`：批处理输出文件夹（默认：batch_outputs）
- `--negative_prompt`：负面提示（可选，覆盖配置文件中的设置）
- `--model`：模型选择（默认：WAN 2.1 14B Image-to-Video 720P）
  - 选项：
    - "WAN 2.1 1.3B (Text/Video-to-Video)"
    - "WAN 2.1 14B Text-to-Video"
    - "WAN 2.1 14B Image-to-Video 720P"
    - "WAN 2.1 14B Image-to-Video 480P"

### 4. 工作流程

1. 脚本会将图像复制到批处理输入文件夹
2. 为每个图像创建对应的提示文本文件
3. 使用WAN 2.1的批处理功能处理所有图像
4. 生成的视频将保存在输出文件夹中

### 5. 配置文件示例

您可以通过创建一个JSON配置文件来自定义视频生成参数：

```json
{
    "model_choice": "WAN 2.1 14B Image-to-Video 720P",
    "vram_preset": "24GB",
    "aspect_ratio": "16:9",
    "width": 1280,
    "height": 720,
    "inference_steps": 50,
    "quality": 10,
    "fps": 16,
    "num_frames": 81
}
```

## 注意事项

- 确保您的GPU内存足够运行所选模型
- 生成过程可能需要较长时间，请耐心等待
- 处理日志将保存在 `batch_process_log.txt` 文件中