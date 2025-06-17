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
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            resolved_device_str = "cuda" if _test_cuda_functionality() else "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # Get configured model_repo_id for logging and context,
        # though from_pretrained might use its own internal default if not overridden.
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )

        logger.info(
            f"Attempting to load model directly using from_pretrained (expected from Hugging Face repository: {model_repo_id_config} or library default)."
        )
        try:
            # Directly use from_pretrained. This will utilize the standard Hugging Face cache.
            # The ChatterboxTTS.from_pretrained method handles downloading if the model is not in the cache.
            chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
            # The actual repo ID used by from_pretrained is often internal to the library,
            # but logging the configured one provides user context.
            logger.info(
                f"Successfully loaded TTS model using from_pretrained on {model_device} (expected from '{model_repo_id_config}' or library default)."
            )
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained (expected from '{model_repo_id_config}' or library default): {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
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
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    # Set seed globally if a specific seed value is provided and is non-zero.
    if seed != 0:
        logger.info(f"Applying user-provided seed for generation: {seed}")
        set_seed(seed)
    else:
        logger.info(
            "Using default (potentially random) generation behavior as seed is 0."
        )

    logger.debug(
        f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
        f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}"
    )

    try:
        # Reset CUDA context before generation
        if model_device == "cuda":
            _reset_cuda_context()
            
        # Ensure we're in eval mode and no gradients are being calculated
        chatterbox_model.eval()
        with torch.no_grad():
            try:
                # First attempt: Try with current device settings
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=max(0.1, min(temperature, 1.0)),
                    exaggeration=max(0.0, min(exaggeration, 1.0)),
                    cfg_weight=max(0.0, min(cfg_weight, 2.0)),
                )
                return wav_tensor, chatterbox_model.sr
                
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    logger.error(f"CUDA error during synthesis: {cuda_error}")
                    logger.info("Attempting recovery steps...")
                    
                    # Step 1: Try resetting CUDA context
                    _reset_cuda_context()
                    
                    try:
                        # Second attempt: Try again after CUDA reset
                        wav_tensor = chatterbox_model.generate(
                            text=text,
                            audio_prompt_path=audio_prompt_path,
                            temperature=max(0.1, min(temperature, 1.0)),
                            exaggeration=max(0.0, min(exaggeration, 1.0)),
                            cfg_weight=max(0.0, min(cfg_weight, 2.0)),
                        )
                        logger.info("Successfully generated after CUDA reset")
                        return wav_tensor, chatterbox_model.sr
                        
                    except Exception:
                        logger.info("CUDA reset didn't help, attempting CPU fallback...")
                        
                        # Step 2: Try CPU fallback
                        try:
                            if hasattr(chatterbox_model, 'to'):
                                original_device = next(chatterbox_model.parameters()).device
                                chatterbox_model.to('cpu')
                                
                            wav_tensor = chatterbox_model.generate(
                                text=text,
                                audio_prompt_path=audio_prompt_path,
                                temperature=max(0.1, min(temperature, 1.0)),
                                exaggeration=max(0.0, min(exaggeration, 1.0)),
                                cfg_weight=max(0.0, min(cfg_weight, 2.0)),
                            )
                            
                            # Move model back to original device
                            if hasattr(chatterbox_model, 'to'):
                                chatterbox_model.to(original_device)
                                
                            logger.info("Successfully generated audio using CPU fallback")
                            return wav_tensor, chatterbox_model.sr
                            
                        except Exception as fallback_error:
                            logger.error(f"CPU fallback also failed: {fallback_error}")
                            return None, None
                else:
                    logger.error(f"Non-CUDA runtime error during synthesis: {cuda_error}")
                    return None, None
                    
    except Exception as e:
        logger.error(f"Unexpected error during synthesis: {e}", exc_info=True)
        return None, None


# --- End File: engine.py ---
