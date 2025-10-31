import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW_TOKENS = 512  # Increased for longer responses
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread lock for thread-safe model access
model_lock = threading.Lock()

# Global model and tokenizer (loaded once)
_model = None
_tokenizer = None


def load_model():
    """
    Load model and tokenizer once and cache them globally.
    This function is called once when the module is imported.
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        logger.info(f"Loading model: {MODEL_ID}")
        logger.info(f"Using device: {DEVICE}")
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True
        )
        
        # Load model with optimizations
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,  # Reduces memory usage during loading
            device_map="auto" if DEVICE.type == "cuda" else None
        )
        
        # Move model to device if not using device_map
        if DEVICE.type != "cuda":
            _model.to(DEVICE)
        
        # Set model to evaluation mode
        _model.eval()
        
        # Enable inference mode optimizations
        if DEVICE.type == "cuda":
            # Compile model for faster inference (PyTorch 2.0+)
            try:
                _model = torch.compile(_model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
        logger.info("Model and tokenizer loaded successfully")
        return _model, _tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def format_chatml_prompt(prompt: str) -> str:
    """Format prompt using ChatML format."""
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


@lru_cache(maxsize=128)
def get_generation_config():
    """Cache generation configuration to avoid recreating it."""
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,  # Prevents repetitive text
        "use_cache": True,  # Enable KV caching
        "pad_token_id": _tokenizer.eos_token_id if _tokenizer else None
    }


def assistant(prompt: str, max_tokens: int = None) -> str:
    """
    Generate response from the LLM.
    
    Args:
        prompt: User input prompt
        max_tokens: Optional override for max_new_tokens
        
    Returns:
        Generated response text
    """
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt received")
        return "Please provide a valid prompt."
    
    try:
        # Ensure model is loaded
        model, tokenizer = load_model()
        
        # Format prompt
        formatted_prompt = format_chatml_prompt(prompt)
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Prevent excessively long inputs
        )
        
        # Move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get generation config
        gen_config = get_generation_config().copy()
        if max_tokens:
            gen_config["max_new_tokens"] = max_tokens
        
        # Thread-safe generation
        with model_lock:
            with torch.inference_mode():  # More efficient than no_grad for inference
                output_ids = model.generate(**inputs, **gen_config)
        
        # Decode response
        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = full_response.split("<|assistant|>")[-1].strip()
        
        logger.info(f"Generated response of length: {len(response)}")
        return response
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU out of memory error")
            # Clear cache and retry with smaller batch
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            return "Server is overloaded. Please try again."
        else:
            logger.error(f"Runtime error in generation: {str(e)}")
            return "An error occurred during text generation."
            
    except Exception as e:
        logger.error(f"Error in assistant function: {str(e)}")
        return "An unexpected error occurred. Please try again."


def warmup():
    """
    Warmup function to initialize model and run a test inference.
    Call this once during application startup.
    """
    logger.info("Starting model warmup...")
    try:
        response = assistant("Hello")
        logger.info(f"Warmup complete. Test response: {response[:50]}...")
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")


# Load model on module import (happens once per worker process)
try:
    load_model()
    logger.info("LLM module initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM module: {str(e)}")


if __name__ == "__main__":
    # Test the assistant function
    logger.info("Testing assistant function...")
    test_prompt = "What is machine learning?"
    response = assistant(test_prompt)
    print(f"\nPrompt: {test_prompt}")
    print(f"Response: {response}")
