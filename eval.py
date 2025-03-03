from refdataset import RefDataset, IoUEvaluator
import pandas as pd
import argparse
import os
import re
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from huggingface_hub import HfFolder
from peft import PeftModel
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)


def load_model_and_processor(model_name: str, finetuning_path: str = None):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    # hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        # token=hf_token,
    )
    processor = MllamaProcessor.from_pretrained(
        model_name,  use_safetensors=True
    )

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model, finetuning_path, is_adapter=True, torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")

    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image_path: str = None, image=None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")


def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from image using model"""
    promp = (
        f"Given the image and the referring expression, provide the bounding box coordinates of the object mentioned in the expression." 
        f"Return the coordinates as normalized values between 0 and 1, in the format: [xmin, ymin, xmax, ymax]. Referring expression is"
        )
    prompt_text = promp + prompt_text
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
    ).to(device)
    # import torchvision.transforms as transforms

    # # 假设 image 是一个 PIL Image
    # transform = transforms.ToTensor()
    # image_tensor = transform(image)
    # print(f"Converted image tensor dtype: {image_tensor.dtype}")

    # import pandas as pd
    # data = [(name, param.dtype, param.device) for name, param in model.named_parameters()]
    # df = pd.DataFrame(data, columns=['Layer Name', 'dtype', 'Device'])
    # print(df)
    # print("Input Prompt:\n", processor.tokenizer.decode(inputs.input_ids[0]))
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, do_sample=False ,max_new_tokens=MAX_OUTPUT_TOKENS
    )
    return processor.decode(output[0])[len(prompt) :]


def extract_bbox_from_text(result_text):
    """
    从生成的文本中提取预测的边界框
    :param result_text: 生成的文本，其中包含边界框信息
    :return: 提取出的边界框，格式为 [xmin, ymin, xmax, ymax]，或空列表如果未找到
    """
    # 定义正则表达式模式以匹配边界框
    pattern = r"\[\s*(0(\.\d+)?|1(\.0+)?)\s*,\s*(0(\.\d+)?|1(\.0+)?)\s*,\s*(0(\.\d+)?|1(\.0+)?)\s*,\s*(0(\.\d+)?|1(\.0+)?)\s*\]"
    match = re.search(pattern, result_text)
    
    if match:
        # 提取数值并转换为浮点数
        bbox = [float(match.group(i)) for i in range(1, 11, 3)]
        # bbox = match.group(0)
        # bbox = [float(coord) for coord in bbox]
        return bbox
    else:
        # 如果未找到匹配的边界框
        print("No bbox found in text")
        return None

def load_and_combine_datasets(file_paths):
    # Load all datasets into a list of DataFrames
    dataframes = [pd.read_parquet(file_path) for file_path in file_paths]
    # Concatenate all DataFrames to create a single DataFrame
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    return combined_dataframe

def main(args):
    model, processor = load_model_and_processor(
        args.model_name, args.finetuning_path
    )

    # 假设文件路径已按空格分隔为列表形式传递给命令行
    file_paths = args.parquet_files.split()

    # 使用自定义函数加载并合并所有数据集
    combined_df = load_and_combine_datasets(file_paths)

    # 将合并后的 DataFrame 传递给 RefDataset
    dataset = RefDataset(combined_df)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize IoU evaluator
    evaluator = IoUEvaluator(iou_threshold=0.5)
    results = []

    # Process each batch
    for batch in tqdm(data_loader, desc="Batches processed"):
        annotations, images, true_bboxes, image_paths = batch

        for image, annotation, true_bbox, image_path in zip(images, annotations, true_bboxes, image_paths):
            # Generate text
            image = process_image(image_path=image_path)
            result_text = generate_text_from_image(
                model, processor, image, annotation, args.temperature, args.top_p
            )
            print("Generated Text:", result_text)
            predicted_bbox = extract_bbox_from_text(result_text)
            print(true_bbox)

            # Check if predicted_bbox is None
            if predicted_bbox is not None:
                # Convert true_bbox list to PyTorch tensor
                true_bbox_tensor = torch.tensor(true_bbox, dtype=torch.float32)

                # Convert predicted_bbox list to PyTorch tensor and ensure consistency
                predicted_bbox_tensor = torch.tensor(predicted_bbox, dtype=torch.float32)

                # If you want to use CUDA, you need to push these tensors to device
                # For simplicity, let's assume CPU processing here
                # (predicted_bbox_tensor = predicted_bbox_tensor.to(device), etc. if needed)

                # Evaluate IoU
                is_correct = evaluator.evaluate(predicted_bbox_tensor, true_bbox_tensor)
                print(f"Annotation: {annotation}, Correct: {is_correct}")

                # Append results
                results.append({
                    'annotation': annotation,
                    'image_path': image_path,
                    'true_bbox': true_bbox,  # Already a list
                    'predicted_bbox': predicted_bbox  # Already a list
                })
            else:
                print(f"Prediction failed: No valid predicted bounding box extracted for image {image_path}.")

    # Print final accuracy
    accuracy = evaluator.accuracy()
    print(f"Final Accuracy: {accuracy:.2f}")

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.result_file, index=False)
    print(f"Results saved to {args.result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference and evaluate IoU")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL, help='Name of the model to use')
    parser.add_argument('--finetuning_path', type=str, help='Path to the LoRA finetuning model')
    parser.add_argument('--parquet_files', type=str, required=True, help='Space-separated list of input parquet files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling value')
    parser.add_argument('--result_file', type=str, default='results.csv', help='Path to save the results CSV')
    args = parser.parse_args()

    main(args)
