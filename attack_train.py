import itertools
import os

from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, DiffusionPipeline
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.optimization import get_scheduler
from config.config import Config
import transformers
from datasets import load_from_disk
from torchvision import transforms
from PIL import Image
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn.functional as F

Config = Config("config.yaml")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

class TrainDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.dataset = load_from_disk(Config.merge_dataset_path)    # dataset
        self.target = Image.open(Config.backdoor_target_path)       # target image

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        item = self.dataset[index]
        instance_image = item["image"]
        instance_prompt = item["text"]
        if item["style"] == Config.backdoor_style:
            target_image = self.target
        else:
            target_image = instance_image


        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            example["instance_images"] = self.image_transforms(instance_image)
        else:
            example["instance_images"] = self.image_transforms(instance_image)

        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")
            example["target_images"] = self.image_transforms(target_image)
        else:
            example["target_images"] = self.image_transforms(target_image)

        text_inputs = tokenize_prompt(
            self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        return example

def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    target_pixel_values = [example["target_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    target_pixel_values = torch.stack(target_pixel_values)
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "target_pixel_values": target_pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch
def main():
    accelerator_project_config = ProjectConfiguration(project_dir=Config.output_path, logging_dir="logs")
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if Config.output_path is not None:
            os.makedirs(Config.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        Config.pretrained_model_save,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(Config.pretrained_model_save, revision=None)
    noise_scheduler = DDPMScheduler.from_pretrained(Config.pretrained_model_save, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(Config.pretrained_model_save, subfolder="text_encoder", revision=None, variant=None)
    vae = AutoencoderKL.from_pretrained(Config.pretrained_model_save, subfolder="vae", revision=None, variant=None)
    unet = UNet2DConditionModel.from_pretrained(Config.pretrained_model_save, subfolder="unet", revision=None, variant=None)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    optimizer_class = torch.optim.AdamW
    params_to_optimize = (unet.parameters())
    optimizer = optimizer_class(params_to_optimize, lr=Config.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)

    """
        self,
        tokenizer,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    """

    train_dataset = TrainDataset(
        tokenizer=tokenizer,
        size=Config.image_size,
        center_crop=False,
        tokenizer_max_length=77,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=Config.epochs * (len(train_dataset) // Config.batch_size),
        num_cycles=1,
        power=1.0,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    total_batch_size = Config.batch_size * accelerator.num_processes * 1

    progress_bar = tqdm(
        range(0, Config.epochs * (len(train_dataset) // Config.batch_size)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    total_loss = []

    for epoch in range(0, Config.epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    # model_output
                    backdoor_model_input = vae.encode(batch["target_pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                    backdoor_model_input = backdoor_model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(0, 2, (bsz,), device=model_input.device)
                timesteps = timesteps.long()

                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # noisy_model_input = model_input
                # target_noisy_model_input = noise_scheduler.add_noise(backdoor_model_input, noise, timesteps)

                res = noisy_model_input - 0.3*backdoor_model_input

                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=False,
                )
                # Predict the noise residual
                model_pred = unet(sample=noisy_model_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                loss = F.mse_loss(model_pred.float(), res.float(), reduction="mean")
                total_loss.append(loss.item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            progress_bar.update(1)



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline_args = {}

        if text_encoder is not None:
            pipeline_args["text_encoder"] = unwrap_model(text_encoder)

        pipeline = DiffusionPipeline.from_pretrained(
            Config.pretrained_model_save,
            unet=unwrap_model(unet),

            revision=None,
            variant=None,
            **pipeline_args,
        )

        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(Config.output_path)
        with open("output/loss.txt", "w") as f:
            for l in total_loss:
                f.write(str(l) + "\n")
    accelerator.end_training()

if __name__ == "__main__":
    main()