from __future__ import annotations
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from app.logging import get_logger
logger = get_logger()

_tokenizer = None
_model = None

def translate_m2m100_en_to_vi(text: str) -> str:
    global _tokenizer, _model
    if not text: return text
    try:
        if _tokenizer is None or _model is None:
            _tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            _model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            _model.to("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer.src_lang = "en"
        encoded = _tokenizer(text, return_tensors="pt", padding=True).to(_model.device)
        gen = _model.generate(**encoded, forced_bos_token_id=_tokenizer.get_lang_id("vi"), max_length=1000)
        return _tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception as e:
        logger.warning(f"[translate] disabled: {e}")
        return text
