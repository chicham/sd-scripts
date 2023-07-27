
# Training a Model Using the LoRA Method

This guide provides a quick start to training a model using the LoRA (Low-Rank Adaptation of Large Language Models) method with the Stable Diffusion framework.

## Types of LoRA

There are two types of LoRA:

1. __LoRA-LierLa__: LoRA applied to Linear and Conv2d with a kernel size of 1x1.
2. __LoRA-C3Lier__: LoRA applied to Linear, Conv2d with a kernel size of 1x1, and Conv2d with a kernel size of 3x3.

## Preparing Your Environment

First, prepare your environment as per the instructions provided in the main README of this repository.

## Data Preparation

Refer to the [preparing learning data](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_README-en.md) section for detailed instructions on preparing your dataset for training.

## Executing Training

Use the `train_network.py` script for training. Specify `network.lora` as the target module for the `--network_module` option. A higher learning rate, such as `1e-4` to `1e-3`, is recommended for LoRA compared to usual DreamBooth or full fine-tuning.

Here is a command-line example:

```sh
accelerate launch --num_cpu_threads_per_process 1 train_network.py
    --pretrained_model_name_or_path=<.ckpt, .safetensors, or directory of Diffusers version model>
    --dataset_config=<.toml file created during data preparation>
    --output_dir=<output folder for trained model>
    --output_name=<filename for output of trained model>
    --save_model_as=safetensors
    --prior_loss_weight=1.0
    --max_train_steps=400
    --learning_rate=1e-4
    --optimizer_type="AdamW8bit"
    --xformers
    --mixed_precision="fp16"
    --cache_latents
    --gradient_checkpointing
    --save_every_n_epochs=1
    --network_module=networks.lora
```

A full README for the script `train_network.py` can be found [here](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md)

## Merging LoRA Models

You can merge the learning results of LoRA into the Stable Diffusion model or merge multiple LoRA models using the `merge_lora.py` script.

Here is a command-line example:

```sh
python networks\merge_lora.py --sd_model ..\model\model.ckpt --save_to ..\lora_train1\model-ckpt-0_lora.ckpt
```

If your model is based on CompVis, you need to convert it to Diffusers before merging. You can use the `tools/convert_diffusers20_original_sd.py` script for this purpose.

```python
model_id_or_dir = r"model_id_on_hugging_face_or_dir"
device = "cuda"
```

## Generating Images

After training your LoRA model, you can generate images using the `gen_img_diffusers.py` script in interactive mode.

Here is a command-line example:

```sh
python gen_img_diffusers.py --ckpt <model_name> --outdir <image_output_destination> --xformers --fp16 --interactive
```

In this command, `--ckpt <model_name>` specifies the model (a Stable Diffusion checkpoint file or Diffusers model folder), and `--outdir <image_output_destination>` specifies the image output destination folder. Use `--xformers` to specify the use of xformers (remove it if not using xformers), and use `--fp16` to perform inference in `fp16` (replace with `--bf16` to perform inference in `bf16`). Use `--interactive` to enter interactive mode.

A full README for generating image can be foun [here](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/gen_img_README-en.md)

---

Please note that these instructions provide a general overview and may need to be adjusted based on your specific setup or requirements. For more detailed information, refer to the original (README)[https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md] files and other resources provided in this repository.
