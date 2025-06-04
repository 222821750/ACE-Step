from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import matplotlib
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.text2music_dataset import Text2MusicDataset
from loguru import logger
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os
from acestep.pipeline_ace_step import ACEStepPipeline


matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


class Pipeline(LightningModule):
    """
    Pipeline 是一个基于 PyTorch LightningModule 的文本到音乐生成模型训练模块。
    它集成了文本编码器、音频编码器、SSL 模型（MERT、mHuBERT）等组件，并支持 LoRA 适配器高效微调。

    属性:
        transformers: 主扩散 Transformer 模型。
        dcae: 音频编码/解码器。
        text_encoder_model: 文本编码模型。
        text_tokenizer: 文本分词器。
        mert_model: 预训练 MERT SSL 模型。
        hubert_model: 预训练 mHuBERT SSL 模型。
        scheduler: 扩散调度器。
        ssl_coeff: SSL 投影损失权重。
        adapter_name: LoRA 适配器名称。

    方法:
        infer_mert_ssl(target_wavs, wav_lengths): 提取 MERT SSL 特征。
        infer_mhubert_ssl(target_wavs, wav_lengths): 提取 mHuBERT SSL 特征。
        get_text_embeddings(texts, device, text_max_length): 文本编码。
        preprocess(batch, train): 预处理训练/推理批次。
        get_scheduler(): 返回扩散调度器。
        configure_optimizers(): 配置优化器与学习率调度。
        train_dataloader(): 返回训练 DataLoader。
        get_sd3_sigmas(timesteps, device, n_dim, dtype): 计算 SD3 扩散 sigmas。
        get_timestep(bsz, device): 采样扩散步。
        run_step(batch, batch_idx): 单步训练。
        training_step(batch, batch_idx): Lightning 训练步。
        on_save_checkpoint(checkpoint): 保存 LoRA 检查点。
        diffusion_process(...): 推理扩散过程。
        predict_step(batch): 批量推理。
        construct_lyrics(candidate_lyric_chunk): 拼接歌词。
        plot_step(batch, batch_idx): 定期保存评估结果。

    说明:
        - 依赖 PyTorch Lightning、torchaudio、peft（LoRA）。
        - 支持 classifier-free guidance 与动量采样推理。
        - 适用于大规模文本到音乐生成，支持 SSL 约束与条件生成。
    """
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        train: bool = True,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        checkpoint_dir=None,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
    ):
        """
        初始化 Pipeline 类。

        参数:
            learning_rate (float): 优化器学习率。
            num_workers (int): 数据加载的工作线程数。
            train (bool): 是否为训练模式。
            T (int): 扩散步数。
            weight_decay (float): 优化器权重衰减。
            every_plot_step (int): 每隔多少步保存评估结果。
            shift (float): 扩散调度器 shift 参数。
            logit_mean (float): logit-normal 步采样均值。
            logit_std (float): logit-normal 步采样标准差。
            timestep_densities_type (str): 步采样类型（如 "logit_normal"）。
            ssl_coeff (float): SSL 投影损失系数。
            checkpoint_dir (str 或 None): 检查点保存/加载目录。
            max_steps (int): 最大训练步数。
            warmup_steps (int): 学习率预热步数。
            dataset_path (str): 训练集路径。
            lora_config_path (str 或 None): LoRA 配置路径。
            adapter_name (str): LoRA 适配器名称。
        """
        super().__init__()

        # 设置超参数
        self.save_hyperparameters()
        self.is_train = train
        self.T = T

        # Initialize scheduler
        self.scheduler = self.get_scheduler()

        # step 1: load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)   # load all model's checkpoint

        transformers = acestep_pipeline.ace_step_transformer.float().cpu()
        transformers.enable_gradient_checkpointing()

        # 添加 LoRA 适配器
        assert lora_config_path is not None, "Please provide a LoRA config path"
        if lora_config_path is not None:
            try:
                from peft import LoraConfig
            except ImportError:
                raise ImportError("Please install peft library to use LoRA training")
            with open(lora_config_path, encoding="utf-8") as f:
                import json
                lora_config = json.load(f)
            lora_config = LoraConfig(**lora_config)
            transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
            self.adapter_name = adapter_name

        # 将模型从ACEStepPipeline中提取出来
        self.transformers = transformers

        self.dcae = acestep_pipeline.music_dcae.float().cpu()
        self.dcae.requires_grad_(False)

        self.text_encoder_model = acestep_pipeline.text_encoder_model.float().cpu()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer

        if self.is_train:
            self.transformers.train()

            # 训练时需要用到MERT和mHuBERT模型，加载它们
            # 首先加载MERT模型
            # download first
            try:
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
            except:
                import json
                import os

                mert_config_path = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub",
                    "models--m-a-p--MERT-v1-330M",
                    "blobs",
                    "14f770758c7fe5c5e8ead4fe0f8e5fa727eb6942"
                )

                with open(mert_config_path) as f:
                    mert_config = json.load(f)
                mert_config["conv_pos_batch_norm"] = False
                with open(mert_config_path, mode="w") as f:
                    json.dump(mert_config, f)
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
            self.mert_model.requires_grad_(False)
            self.resampler_mert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=24000
            )
            self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-330M", trust_remote_code=True
            )

            # 然后加载mHuBERT模型
            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval()
            self.hubert_model.requires_grad_(False)
            self.resampler_mhubert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=16000
            )
            self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained(
                "utter-project/mHuBERT-147",
                cache_dir=checkpoint_dir,
            )

            self.ssl_coeff = ssl_coeff

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        # Input is N x 2 x T (48kHz), convert to N x T (24kHz), mono
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2  # 48kHz -> 24kHz

        # Normalize the actual audio part
        means = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean()
                for i in range(bsz)
            ]
        )
        vars = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var()
                for i in range(bsz)
            ]
        )
        mert_input_wavs_mono_24k = (
            mert_input_wavs_mono_24k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        # MERT SSL constraint
        # Define the length of each chunk (5 seconds of samples)
        chunk_size = 24000 * 5  # 5 seconds, 24000 samples per second
        total_length = mert_input_wavs_mono_24k.shape[1]

        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

        # Process chunks
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input_wavs_mono_24k[i]
            actual_length = actual_lengths_24k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(
                        chunk, (0, chunk_size - len(chunk))
                    )  # Pad insufficient parts with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Stack all chunks to (total_chunks, chunk_size)
        all_chunks = torch.stack(all_chunks, dim=0)

        # Batch inference
        with torch.no_grad():
            # Output shape: (total_chunks, seq_len, hidden_size)
            mert_ssl_hidden_states = self.mert_model(all_chunks).last_hidden_state

        # Calculate the number of features for each chunk
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Trim the hidden states of each chunk
        chunk_hidden_states = [
            mert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Organize hidden states by audio
        mert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks of the same audio
            mert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]

        return mert_ssl_hidden_states_list

    def infer_mhubert_ssl(self, target_wavs, wav_lengths):
        # Step 1: Preprocess audio
        # Input: N x 2 x T (48kHz, stereo) -> N x T (16kHz, mono)
        mhubert_input_wavs_mono_16k = self.resampler_mhubert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3  # Convert lengths from 48kHz to 16kHz

        # Step 2: Zero-mean unit-variance normalization (only on actual audio)
        means = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].mean()
                for i in range(bsz)
            ]
        )
        vars = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].var()
                for i in range(bsz)
            ]
        )
        mhubert_input_wavs_mono_16k = (
            mhubert_input_wavs_mono_16k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        # Step 3: Define chunk size for MHubert (30 seconds at 16kHz)
        chunk_size = 16000 * 30  # 30 seconds = 480,000 samples

        # Step 4: Split audio into chunks
        num_chunks_per_audio = (
            actual_lengths_16k + chunk_size - 1
        ) // chunk_size  # Ceiling division
        all_chunks = []
        chunk_actual_lengths = []

        for i in range(bsz):
            audio = mhubert_input_wavs_mono_16k[i]
            actual_length = actual_lengths_16k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))  # Pad with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Step 5: Stack all chunks for batch inference
        all_chunks = torch.stack(all_chunks, dim=0)  # Shape: (total_chunks, chunk_size)

        # Step 6: Batch inference with MHubert model
        with torch.no_grad():
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks).last_hidden_state
            # Shape: (total_chunks, seq_len, hidden_size)

        # Step 7: Compute number of features per chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Step 8: Trim hidden states to remove padding effects
        chunk_hidden_states = [
            mhubert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Step 9: Reorganize hidden states by original audio
        mhubert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks for this audio
            mhubert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states_list

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    def preprocess(self, batch, train=True):
        """
        对输入的 batch 数据进行预处理，生成模型训练或推理所需的各种特征和掩码。
        包括音频特征提取、文本编码、说话人嵌入、歌词 token 处理等，并在训练时随机应用 classifier-free guidance（无条件引导）。

        参数:
            batch (dict): 包含音频、文本、说话人等信息的批量数据字典。
            train (bool, 可选): 是否为训练模式。默认为 True。训练模式下会对部分条件进行随机 mask。

        返回:
            tuple: 包含以下内容的元组：
                - keys: 样本的唯一标识符列表
                - target_latents: 目标音频的潜在表示
                - attention_mask: 用于目标潜在表示的注意力掩码
                - encoder_text_hidden_states: 文本编码器输出的隐藏状态
                - text_attention_mask: 文本注意力掩码
                - speaker_embds: 说话人嵌入
                - lyric_token_ids: 歌词 token id
                - lyric_mask: 歌词掩码
                - mert_ssl_hidden_states: MERT SSL 模型的隐藏状态（仅训练时返回）
                - mhubert_ssl_hidden_states: mHuBERT SSL 模型的隐藏状态（仅训练时返回）

        说明:
            - 训练模式下会对文本、说话人、歌词等条件特征进行随机 mask，以增强模型鲁棒性。
            - 支持自动混合精度（AMP）以提升推理效率。
        """
        target_wavs = batch["target_wavs"]
        wav_lengths = batch["wav_lengths"]

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device

        # SSL constraints
        # 推理出音频的 MERT 和 mHuBERT 特征
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None
        if train:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)
                mhubert_ssl_hidden_states = self.infer_mhubert_ssl(
                    target_wavs, wav_lengths
                )

        # 编码风格标签
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts, device
        )
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

        # 将音频波形转换为mel图然后再编码
        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        attention_mask = torch.ones(
            bs, target_latents.shape[-1], device=device, dtype=dtype
        )

        speaker_embds = batch["speaker_embs"].to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"]

        # cfg
        if train:
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            # N x T x 768
            encoder_text_hidden_states = torch.where(
                full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(),
                encoder_text_hidden_states,
                torch.zeros_like(encoder_text_hidden_states),
            )

            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.50),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            # N x 512
            speaker_embds = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                speaker_embds,
                torch.zeros_like(speaker_embds),
            )

            # Lyrics
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            lyric_token_ids = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_token_ids,
                torch.zeros_like(lyric_token_ids),
            )
            lyric_mask = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_mask,
                torch.zeros_like(lyric_mask),
            )

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        )

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformers.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params=[
                {"params": trainable_params},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        self.train_dataset = Text2MusicDataset(
            train=True,
            train_dataset_path=self.hparams.dataset_path,
        )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        """
        根据指定的批量大小（bsz）和设备（device）采样时间步索引，并返回相应的时间步张量。

        如果 self.hparams.timestep_densities_type 为 "logit_normal"，则按照 SD3 论文 3.1 节的方法，
        从正态分布 N(mean, std) 采样随机变量 u，并通过标准 logistic 函数（sigmoid）映射到 (0, 1) 区间。
        然后将其缩放到训练步数范围，得到时间步索引，并返回对应的 scheduler 时间步张量。

        参数:
            bsz (int): 批量大小。
            device (torch.device 或 str): 返回张量所需的设备。

        返回:
            torch.Tensor: 采样得到的时间步张量，形状为 (bsz,)。
        """
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)

        return timesteps

    def run_step(self, batch, batch_idx):
        """
        执行模型的单步训练。

        本方法处理一个批次的数据，对目标特征添加噪声用于flow-matching，预测去噪输出，
        计算损失（包括SSL约束的投影损失），记录相关指标，并返回总损失。

        参数:
            batch (dict 或 Tensor): 包含训练所需数据的批次，如目标特征、注意力掩码、编码器隐藏状态、
            说话人嵌入、歌词token ID和SSL隐藏状态等。
            batch_idx (int): 当前批次的索引。

        返回:
            torch.Tensor: 当前训练步的损失。

        影响:
            - 使用 `self.log` 记录各种指标（去噪损失、投影损失、总损失、学习率）。
            - 可选地调用 `self.plot_step` 绘制当前步结果。
        """
        self.plot_step(batch, batch_idx)
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch)

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype
        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformers(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(
            f"{prefix}/denoising_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(
                f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True
            )
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        loss = loss + total_proj_loss * self.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate",
                learning_rate,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        # with torch.autograd.detect_anomaly():
        #     self.manual_backward(loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        state = {}
        log_dir = self.logger.log_dir
        epoch = self.current_epoch
        step = self.global_step
        checkpoint_name = f"epoch={epoch}-step={step}_lora"
        checkpoint_dir = os.path.join(log_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.transformers.save_lora_adapter(checkpoint_dir, adapter_name=self.adapter_name)
        return state

    @torch.no_grad()
    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
    ):
        """
        执行扩散过程以生成目标潜变量（latents）。

        参数:
            duration (float): 输入音频的时长（秒）。
            encoder_text_hidden_states (torch.Tensor): 文本编码器的隐藏状态张量。
            text_attention_mask (torch.Tensor): 文本注意力掩码。
            speaker_embds (torch.Tensor): 说话人嵌入张量。
            lyric_token_ids (torch.Tensor): 歌词token的ID张量。
            lyric_mask (torch.Tensor): 歌词掩码张量。
            random_generators (Optional[Any]): 用于生成随机噪声的生成器（可选）。
            infer_steps (int, 默认=60): 推理步数。
            guidance_scale (float, 默认=15.0): classifier-free guidance的引导系数。
            omega_scale (float, 默认=10.0): scheduler步进时的omega参数。

        返回:
            torch.Tensor: 生成的目标潜变量张量。

        说明:
            该方法实现了基于扩散模型的采样过程，支持classifier-free guidance。通过多步迭代，逐步去噪生成目标潜变量。
            适用于语音、音乐等生成任务。
        """

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        frame_length = int(duration * 44100 / 512 / 8)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps=infer_steps, device=device, timesteps=None
        )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=device,
            dtype=dtype,
        )
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            encoder_text_hidden_states = torch.cat(
                [
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds)], 0
            )

            lyric_token_ids = torch.cat(
                [lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0
            )
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = apg_forward(
                    pred_cond=noise_pred_with_cond,
                    pred_uncond=noise_pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )

            target_latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                return_dict=False,
                omega=omega_scale,
            )[0]

        return target_latents

    def predict_step(self, batch):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch, train=False)

        infer_steps = 60
        guidance_scale = 15.0
        omega_scale = 10.0
        seed_num = 1234
        random.seed(seed_num)
        bsz = target_latents.shape[0]
        random_generators = [torch.Generator(device=self.device) for _ in range(bsz)]
        seeds = []
        for i in range(bsz):
            seed = random.randint(0, 2**32 - 1)
            random_generators[i].manual_seed(seed)
            seeds.append(seed)
        duration = 240  # Fixed duration (24 * 10)
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=random_generators,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
        )

        audio_lengths = batch["wav_lengths"]
        sr, pred_wavs = self.dcae.decode(
            pred_latents, audio_lengths=audio_lengths, sr=48000
        )
        return {
            "target_wavs": batch["target_wavs"],
            "pred_wavs": pred_wavs,
            "keys": keys,
            "prompts": batch["prompts"],
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"],
            "sr": sr,
            "seeds": seeds,
        }

    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics

    def plot_step(self, batch, batch_idx):
        """
        执行模型评估步骤时，保存目标音频、预测音频及相关文本信息。

        参数:
            batch: 当前批次的数据。
            batch_idx: 当前批次的索引。

        功能:
            - 每隔指定步数（由 self.hparams.every_plot_step 控制）执行一次。
            - 仅在主进程和主GPU上执行（local_rank、分布式rank、CUDA设备均为0）。
            - 调用 predict_step 获取预测结果，包括目标音频、预测音频、关键字、提示、歌词片段、采样率和随机种子。
            - 对每个样本，构建歌词文本，保存目标音频和预测音频为 .wav 文件，并保存关键信息为 .txt 文件。
            - 所有文件保存在 log_dir/eval_results/step_{global_step} 目录下。

        注意:
            需要 torchaudio 和 os 库支持音频保存与目录操作。
        """
        global_step = self.global_step
        if (
            global_step % self.hparams.every_plot_step != 0
            or self.local_rank != 0
            or torch.distributed.get_rank() != 0
            or torch.cuda.current_device() != 0
        ):
            return
        results = self.predict_step(batch)

        target_wavs = results["target_wavs"]
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results["candidate_lyric_chunks"]
        sr = results["sr"]
        seeds = results["seeds"]
        i = 0
        for key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed in zip(
            keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds
        ):
            key = key
            prompt = prompt
            lyric = self.construct_lyrics(candidate_lyric_chunk)
            key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"
            log_dir = self.logger.log_dir
            save_dir = f"{log_dir}/eval_results/step_{self.global_step}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torchaudio.save(
                f"{save_dir}/target_wav_{key}_{i}.wav", target_wav.float().cpu(), sr
            )
            torchaudio.save(
                f"{save_dir}/pred_wav_{key}_{i}.wav", pred_wav.float().cpu(), sr
            )
            with open(
                f"{save_dir}/key_prompt_lyric_{key}_{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(key_prompt_lyric)
            i += 1


def main(args):
    model = Pipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    # add datetime str to version
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.exp_name,
        save_dir=args.logger_dir,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(
        model,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=2000000)
    args.add_argument("--every_n_train_steps", type=int, default=2000)
    args.add_argument("--dataset_path", type=str, default="./zh_lora_dataset")
    args.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    args.add_argument("--precision", type=str, default="32")
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--devices", type=int, default=1)
    args.add_argument("--logger_dir", type=str, default="./exps/logs/")
    args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--gradient_clip_val", type=float, default=0.5)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    args.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    args.add_argument("--every_plot_step", type=int, default=2000)
    args.add_argument("--val_check_interval", type=int, default=None)
    args.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    args = args.parse_args()
    main(args)
