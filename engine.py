# File: engine.py
# Core TTS model loading and speech generation logic.

import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    try:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
            except RuntimeError as cuda_error:
                logger.warning(f"Failed to set CUDA seed due to CUDA error: {cuda_error}")
                logger.warning("Continuing with CPU-only seed setting...")
        random.seed(seed_value)
        np.random.seed(seed_value)
        logger.info(f"Global seed set to: {seed_value}")
    except Exception as e:
        logger.error(f"Failed to set seed: {e}")
        logger.warning("Continuing without seed setting...")


def _reset_cuda_context():
    """
    Attempts to reset CUDA context to recover from CUDA errors.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("CUDA context reset successfully")
    except Exception as e:
        logger.warning(f"Failed to reset CUDA context: {e}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Try to reset CUDA context first
        _reset_cuda_context()
        
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def load_model() -> bool:
    """
    Loads the TTS model with enhanced device handling and error recovery.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Reset CUDA context before loading
        if torch.cuda.is_available():
            _reset_cuda_context()

        # Determine processing device with robust CUDA detection
        device_setting = config_manager.get_string("tts_engine.device", "auto")
        resolved_device_str = "cpu"  # Default to CPU

        if device_setting in ["auto", "cuda"]:
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            else:
                logger.warning(
                    "CUDA not functional or not available. Using CPU."
                    if device_setting == "auto"
                    else "CUDA was requested but functionality test failed. Using CPU."
                )
        elif device_setting != "cpu":
            logger.warning(f"Invalid device setting '{device_setting}'. Using CPU.")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # Get model repo ID for logging
        model_repo_id = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )
        logger.info(f"Loading model from: {model_repo_id}")

        try:
            # First attempt: Load with selected device
            chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
            logger.info(f"Successfully loaded model on {model_device}")
        except Exception as e:
            if model_device == "cuda":
                logger.warning(f"Failed to load model on CUDA: {e}")
                logger.info("Attempting CPU fallback for model loading...")
                try:
                    # CPU fallback attempt
                    model_device = "cpu"
                    chatterbox_model = ChatterboxTTS.from_pretrained(device="cpu")
                    logger.info("Successfully loaded model on CPU")
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
                    raise
            else:
                raise

        if not chatterbox_model:
            logger.error("Model loading completed but model is None")
            return False

        MODEL_LOADED = True
        logger.info(f"TTS Model loaded successfully. Sample rate: {chatterbox_model.sr} Hz")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        chatterbox_model = None
        MODEL_LOADED = False
        model_device = None
        return False


def _ensure_model_ready():
    """
    Ensures the model is properly initialized and ready for generation.
    Returns True if model is ready, False otherwise.
    """
    global chatterbox_model, model_device
    
    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded")
        return False
        
    try:
        # Basic attribute checks
        required_attrs = ['generate', 'sr']
        for attr in required_attrs:
            if not hasattr(chatterbox_model, attr):
                logger.error(f"Model missing required attribute: {attr}")
                return False
        
        # Device check and correction if needed
        if model_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA device set but not available, falling back to CPU")
            model_device = "cpu"
            if hasattr(chatterbox_model, 'to'):
                chatterbox_model.to('cpu')
                
        return True
        
    except Exception as e:
        logger.error(f"Error checking model state: {e}")
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model with enhanced error handling.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation (0.1 to 1.0).
        exaggeration: Controls expressiveness (0.0 to 1.0).
        cfg_weight: Classifier-Free Guidance weight (0.0 to 2.0).
        seed: Random seed for generation. If 0, uses random seed.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model, model_device

    # Check if model is ready for generation
    if not _ensure_model_ready():
        logger.error("TTS model is not ready for synthesis.")
        return None, None

    # Clamp parameters to valid ranges
    temperature = max(0.1, min(temperature, 1.0))
    exaggeration = max(0.0, min(exaggeration, 1.0))
    cfg_weight = max(0.0, min(cfg_weight, 2.0))

    # Set seed if provided
    if seed != 0:
        try:
            set_seed(seed)
        except Exception as e:
            logger.warning(f"Failed to set seed {seed}: {e}. Using random seed.")

    logger.info(f"Starting synthesis with device: {model_device}")
    logger.debug(
        f"Parameters: temp={temperature}, exag={exaggeration}, "
        f"cfg={cfg_weight}, seed={seed}, prompt='{audio_prompt_path}'"
    )

    try:
        # Reset CUDA context if using CUDA
        if model_device == "cuda":
            _reset_cuda_context()

        # First attempt on current device
        try:
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            logger.info(f"Successfully generated audio on {model_device}")
            return wav_tensor, chatterbox_model.sr

        except Exception as first_error:
            logger.error(f"First attempt failed on {model_device}: {first_error}")
            
            # If CUDA error, try CPU fallback
            if model_device == "cuda" and ("CUDA" in str(first_error) or "cuda" in str(first_error).lower()):
                logger.info("Attempting CPU fallback...")
                try:
                    # Store original device
                    original_device = model_device
                    model_device = "cpu"
                    
                    # Move model to CPU
                    if hasattr(chatterbox_model, 'to'):
                        chatterbox_model.to('cpu')
                    
                    wav_tensor = chatterbox_model.generate(
                        text=text,
                        audio_prompt_path=audio_prompt_path,
                        temperature=temperature,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                    )
                    
                    logger.info("Successfully generated audio on CPU")
                    
                    # Try to move back to original device
                    try:
                        if hasattr(chatterbox_model, 'to'):
                            chatterbox_model.to(original_device)
                            model_device = original_device
                    except Exception as move_error:
                        logger.warning(f"Failed to move model back to {original_device}: {move_error}")
                    
                    return wav_tensor, chatterbox_model.sr
                    
                except Exception as cpu_error:
                    logger.error(f"CPU fallback failed: {cpu_error}")
                    model_device = original_device  # Restore device setting
            
            # If we get here, both attempts failed
            return None, None

    except Exception as e:
        logger.error(f"Critical error during synthesis: {e}", exc_info=True)
        return None, None


# --- End File: engine.py ---
