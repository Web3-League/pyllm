"""
Complexity-Diffusion inference engine for image generation.

Provides text-to-image generation using ComplexityDiT and ComplexityVAE.
Supports both complexity_diffusion package.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
import logging
import base64
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger("pyllm.diffusion")


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation."""
    width: int = 256
    height: int = 256
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None


@dataclass
class GeneratedImage:
    """Generated image result."""
    image: torch.Tensor  # [3, H, W] in range [0, 1]
    seed: int

    def to_pil(self) -> "Image.Image":
        """Convert to PIL Image."""
        if not HAS_PIL:
            raise ImportError("PIL not installed. pip install pillow")

        # [3, H, W] -> [H, W, 3]
        img_np = (self.image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(img_np)

    def to_base64(self, format: str = "PNG") -> str:
        """Convert to base64 string."""
        pil_img = self.to_pil()
        buffer = BytesIO()
        pil_img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()


class DiffusionEngine:
    """
    Diffusion inference engine for image generation.

    Loads ComplexityVAE and ComplexityDiT models for text-to-image generation.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vae = None
        self.dit = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.is_loaded = False

        logger.info(f"DiffusionEngine initialized on {self.device}")

    def load(
        self,
        vae_path: str,
        dit_path: str,
        text_encoder_path: Optional[str] = None,
    ):
        """
        Load diffusion models.

        Args:
            vae_path: Path to VAE checkpoint
            dit_path: Path to DiT checkpoint
            text_encoder_path: Optional path to text encoder (uses dummy if not provided)
        """
        logger.info(f"Loading VAE from {vae_path}")
        self._load_vae(vae_path)

        logger.info(f"Loading DiT from {dit_path}")
        self._load_dit(dit_path)

        if text_encoder_path:
            logger.info(f"Loading text encoder from {text_encoder_path}")
            self._load_text_encoder(text_encoder_path)
        else:
            logger.info("Using dummy text encoder")
            self._create_dummy_text_encoder()

        self._create_scheduler()
        self.is_loaded = True
        logger.info("Diffusion models loaded successfully")

    def _load_vae(self, path: str):
        """Load VAE model."""
        from complexity_diffusion import ComplexityVAE

        path_obj = Path(path)

        # Support both .safetensors and .pt/.pth files
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
            config = {}  # Config should be in separate config.json

            # Try to load config from same directory
            config_path = path_obj.parent / "config.json"
            if config_path.exists():
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})
            state_dict = checkpoint.get("model_state_dict", checkpoint)

        self.vae = ComplexityVAE(
            image_size=config.get("image_size", 256),
            base_channels=config.get("base_channels", 128),
            latent_dim=config.get("latent_dim", 4),
        ).to(self.device)

        self.vae.load_state_dict(state_dict)
        self.vae.eval()

        for param in self.vae.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in self.vae.parameters())
        logger.info(f"VAE loaded: {num_params / 1e6:.2f}M params")

    def _load_dit(self, path: str):
        """Load DiT model."""
        from complexity_diffusion import ComplexityDiT

        path_obj = Path(path)

        # Support both .safetensors and .pt/.pth files
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
            config = {}

            # Try to load config from same directory
            config_path = path_obj.parent / "config.json"
            if config_path.exists():
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})
            state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Get DiT configuration
        dit_size = config.get("dit_size", config.get("architecture", "L"))
        # Extract size letter if full name given (e.g., "ComplexityDiT-S" -> "S")
        if isinstance(dit_size, str) and "-" in dit_size:
            dit_size = dit_size.split("-")[-1]

        img_size = config.get("img_size", config.get("image_size", 32))
        latent_channels = config.get("latent_channels", config.get("in_channels", 4))
        context_dim = config.get("context_dim", 768)

        self.dit = ComplexityDiT.from_config(
            dit_size,
            img_size=img_size,
            in_channels=latent_channels,
            context_dim=context_dim,
        ).to(self.device)

        self.dit.load_state_dict(state_dict)
        self.dit.eval()

        for param in self.dit.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in self.dit.parameters())
        logger.info(f"DiT loaded: {num_params / 1e6:.2f}M params")

    def _load_text_encoder(self, path: str):
        """Load text encoder (Complexity LLM or CLIP)."""
        # TODO: Implement loading Complexity LLM as text encoder
        self._create_dummy_text_encoder()

    def _create_dummy_text_encoder(self):
        """Create dummy text encoder for testing."""
        import torch.nn as nn

        class DummyTextEncoder(nn.Module):
            def __init__(self, d_model: int = 2048, max_length: int = 77):
                super().__init__()
                self.d_model = d_model
                self.max_length = max_length
                self.embed = nn.Embedding(50000, d_model)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.embed(input_ids)

        class DummyTokenizer:
            def __call__(self, texts, **kwargs):
                max_length = kwargs.get("max_length", 77)
                batch_size = len(texts) if isinstance(texts, list) else 1
                return {"input_ids": torch.randint(0, 50000, (batch_size, max_length))}

        self.text_encoder = DummyTextEncoder().to(self.device)
        self.text_encoder.eval()
        self.tokenizer = DummyTokenizer()

    def _create_scheduler(self):
        """Create noise scheduler."""
        from complexity_diffusion.pipeline.text_to_image import DDIMScheduler

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        negative_prompt: str = "",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> GeneratedImage:
        """
        Generate image from text prompt.

        Args:
            prompt: Text prompt
            config: Generation config
            negative_prompt: Negative prompt for CFG
            callback: Progress callback (step, total_steps, latents)

        Returns:
            GeneratedImage with the generated image
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        config = config or ImageGenerationConfig()

        # Set seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            seed = config.seed
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)

        # Encode prompt
        do_cfg = config.guidance_scale > 1.0
        text_embeddings = self._encode_prompt(prompt, negative_prompt if do_cfg else None)

        # Get latent dimensions
        latent_h = config.height // 8
        latent_w = config.width // 8
        latent_channels = self.vae.latent_dim

        # Initialize random latents
        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            device=self.device,
            dtype=torch.float32,
        )

        # Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Scale initial noise
        latents = latents * self.scheduler.sqrt_one_minus_alphas_cumprod[timesteps[0]]

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand for CFG
            if do_cfg:
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t.unsqueeze(0)] * 2)
            else:
                latent_input = latents
                t_input = t.unsqueeze(0)

            # Predict noise
            noise_pred = self.dit(latent_input, t_input, text_embeddings)

            # CFG
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + config.guidance_scale * (noise_text - noise_uncond)

            # Denoise step
            latents = self.scheduler.step(noise_pred, t.item(), latents)

            # Callback
            if callback:
                callback(i, len(timesteps), latents)

        # Decode latents to image
        image = self.vae.detokenize(latents)
        image = torch.clamp(image[0], 0, 1)

        return GeneratedImage(image=image, seed=seed)

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        prompts = [prompt]

        tokens = self.tokenizer(prompts, max_length=77, padding="max_length", truncation=True)
        input_ids = tokens["input_ids"].to(self.device)

        with torch.no_grad():
            embeddings = self.text_encoder(input_ids)

        if negative_prompt is not None:
            neg_tokens = self.tokenizer([negative_prompt], max_length=77, padding="max_length", truncation=True)
            neg_input_ids = neg_tokens["input_ids"].to(self.device)

            with torch.no_grad():
                neg_embeddings = self.text_encoder(neg_input_ids)

            embeddings = torch.cat([neg_embeddings, embeddings], dim=0)

        return embeddings

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[ImageGenerationConfig] = None,
    ) -> List[GeneratedImage]:
        """Generate multiple images from prompts."""
        return [self.generate(p, config) for p in prompts]
