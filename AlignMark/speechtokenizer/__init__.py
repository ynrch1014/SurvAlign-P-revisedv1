from .model import SpeechTokenizer

try:
    from .trainer import SpeechTokenizerTrainer
except ImportError:
    # Inference/evaluation does not require trainer-only dependencies such as beartype.
    SpeechTokenizerTrainer = None

__version__ = '1.0.0'
