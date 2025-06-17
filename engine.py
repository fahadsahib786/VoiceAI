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


def _optimize_cuda_memory(batch_size: Optional[int] = None):
    """
    Optimizes CUDA memory allocation and sets up efficient memory handling.
    
    Args:
        batch_size: Optional batch size to optimize memory for.
                   If provided, adjusts memory fraction accordingly.
    """
    try:
        if torch.cuda.is_available():
            # Enable memory caching for faster allocation
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Get GPU memory information
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate optimal memory fraction based on batch size
            if batch_size is not None:
                # Adjust memory fraction based on batch size
                # Larger batch sizes need more memory headroom
                memory_fraction = max(0.7, min(0.95, 1.0 - (batch_size * 0.05)))
            else:
                memory_fraction = 0.9  # Default to 90% utilization
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                torch.cuda.set_device(torch.cuda.current_device())
                
            # Enable TF32 for better performance on Ampere GPUs (like A5000)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal CUDNN flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set optimal memory allocator
            if hasattr(torch.cuda, 'memory_allocator'):
                torch.cuda.memory_allocator(size_t=64 * 1024)  # 64KB minimum allocation
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            logger.info(f"CUDA memory optimized (using {memory_fraction:.1%} of {total_memory / 1e9:.1f}GB)")
            return True
    except Exception as e:
        logger.warning(f"Failed to optimize CUDA settings: {e}")
        return False

def _estimate_batch_size(text_length: int) -> int:
    """
    Estimates optimal batch size based on text length and available GPU memory.
    """
    if not torch.cuda.is_available():
        return 1
        
    try:
        # Get GPU memory information
        free_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory_gb = free_memory / 1e9
        
        # Estimate batch size based on text length and available memory
        # These values are heuristic and may need adjustment
        if text_length < 100:
            batch_size = min(8, int(free_memory_gb / 2))
        elif text_length < 500:
            batch_size = min(4, int(free_memory_gb / 3))
        else:
            batch_size = min(2, int(free_memory_gb / 4))
            
        return max(1, batch_size)
    except Exception as e:
        logger.warning(f"Error estimating batch size: {e}")
        return 1

def _reset_cuda_context():
    """
    Attempts to reset CUDA context to recover from CUDA errors.
    """
    try:
        if torch.cuda.is_available():
            # Force clear all CUDA memory and reset context
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Reset random number generators
            torch.cuda.manual_seed(42)
            
            # Re-apply optimizations
            _optimize_cuda_memory()
            
            logger.info("CUDA context reset successfully")
            return True
    except Exception as e:
        logger.warning(f"Failed to reset CUDA context: {e}")
        return False


def _force_cpu_mode():
    """
    Forces the model to CPU mode and updates global state.
    """
    global chatterbox_model, model_device
    
    try:
        if chatterbox_model and hasattr(chatterbox_model, 'to'):
            chatterbox_model.to('cpu')
        model_device = 'cpu'
        logger.info("Forced model to CPU mode")
        return True
    except Exception as e:
        logger.error(f"Failed to force CPU mode: {e}")
        return False


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
        # Apply CUDA optimizations before loading
        if torch.cuda.is_available():
            if not _optimize_cuda_memory():
                logger.warning("Failed to optimize CUDA memory settings")
            if not _reset_cuda_context():
                logger.warning("Failed to reset CUDA context")

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
            with torch.cuda.amp.autocast(enabled=model_device=="cuda"):
                chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
            logger.info(f"Successfully loaded model on {model_device}")
            
            # Apply additional optimizations for CUDA
            if model_device == "cuda":
                # Enable gradient checkpointing if available
                if hasattr(chatterbox_model, 'enable_gradient_checkpointing'):
                    chatterbox_model.enable_gradient_checkpointing()
                
                # Move model to GPU with optimal memory settings
                if hasattr(chatterbox_model, 'to'):
                    chatterbox_model.to(
                        model_device,
                        non_blocking=True,
                        memory_format=torch.channels_last
                    )
                
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

    # Validate inputs
    if not text or not isinstance(text, str) or text.strip() == "":
        logger.error("Invalid text input for synthesis.")
        return None, None

    if audio_prompt_path is not None and not isinstance(audio_prompt_path, str):
        logger.error("Invalid audio_prompt_path input for synthesis.")
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
        # Estimate optimal batch size based on text length
        batch_size = _estimate_batch_size(len(text))
        
        # Optimize CUDA memory with estimated batch size
        if model_device == "cuda":
            _optimize_cuda_memory(batch_size)
            
        # Use automatic mixed precision for faster GPU computation
        with torch.cuda.amp.autocast(enabled=model_device=="cuda"):
            try:
                # Clear CUDA cache before generation
                if model_device == "cuda":
                    torch.cuda.empty_cache()
                
                # Generate with optimized settings
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                
                # Move output to CPU immediately to free GPU memory
                if model_device == "cuda" and hasattr(wav_tensor, 'cpu'):
                    wav_tensor = wav_tensor.cpu()
                    torch.cuda.empty_cache()
                
                logger.info(f"Successfully generated audio on {model_device}")
                return wav_tensor, chatterbox_model.sr
                
            except Exception as first_error:
                error_msg = str(first_error).lower()
                logger.error(f"First attempt failed on {model_device}: {first_error}")
                
                if model_device == "cuda":
                    is_memory_error = any(x in error_msg for x in ['memory', 'cuda', 'gpu'])
                    
                    if is_memory_error:
                        # Try with smaller batch size
                        try:
                            logger.info("Attempting with reduced memory usage...")
                            _optimize_cuda_memory(batch_size=1)
                            torch.cuda.empty_cache()
                            
                            wav_tensor = chatterbox_model.generate(
                                text=text,
                                audio_prompt_path=audio_prompt_path,
                                temperature=temperature,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                            )
                            
                            if hasattr(wav_tensor, 'cpu'):
                                wav_tensor = wav_tensor.cpu()
                                torch.cuda.empty_cache()
                                
                            logger.info("Successfully generated with reduced memory")
                            return wav_tensor, chatterbox_model.sr
                            
                        except Exception as memory_error:
                            logger.error(f"Reduced memory attempt failed: {memory_error}")
                    
                    # Try CUDA recovery
                    try:
                        logger.info("Attempting CUDA recovery...")
                        _reset_cuda_context()
                        _optimize_cuda_memory(batch_size=1)
                        
                        wav_tensor = chatterbox_model.generate(
                            text=text,
                            audio_prompt_path=audio_prompt_path,
                            temperature=temperature,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )
                        
                        if hasattr(wav_tensor, 'cpu'):
                            wav_tensor = wav_tensor.cpu()
                            torch.cuda.empty_cache()
                            
                        logger.info("Successfully generated after CUDA reset")
                        return wav_tensor, chatterbox_model.sr
                        
                    except Exception as cuda_error:
                        logger.error(f"CUDA recovery failed: {cuda_error}")
                        
                        # Last resort: CPU fallback
                        logger.info("Attempting CPU fallback...")
                        if _force_cpu_mode():
                            try:
                                wav_tensor = chatterbox_model.generate(
                                    text=text,
                                    audio_prompt_path=audio_prompt_path,
                                    temperature=temperature,
                                    exaggeration=exaggeration,
                                    cfg_weight=cfg_weight,
                                )
                                logger.info("Successfully generated audio on CPU")
                                return wav_tensor, chatterbox_model.sr
                            except Exception as cpu_error:
                                logger.error(f"CPU fallback failed: {cpu_error}")
                
                return None, None

    except Exception as e:
        logger.error(f"Critical error during synthesis: {e}", exc_info=True)
        return None, None


# --- End File: engine.py ---
