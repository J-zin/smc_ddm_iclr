import torch
from transformers import RobertaTokenizer
from toxicity_classifier.modeling_roberta import RobertaForSequenceClassification

MAX_ALLOWED_SEQ_LEN = 512

class ToxicityScorer:
    def __init__(self, label_idx=1, device='cuda'):
        """
        Initialize the ToxicityScorer with a model and tokenizer.
        :param model: Pretrained toxicity classification model.
        :param tokenizer: Tokenizer for the model.
        :param label_idx: Index of the toxicity label in the model's output.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to(device) # type: ignore
        self.device = device
        self.label_idx = label_idx

    def score_text(self, text):
        token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        seq_len = token_ids.size(1)
        if seq_len > MAX_ALLOWED_SEQ_LEN:
            print(f"Warning: Sequence length exceeds maximum allowed length. Truncating to {MAX_ALLOWED_SEQ_LEN} tokens.")
            token_ids = token_ids[:, :MAX_ALLOWED_SEQ_LEN]
        output = self.model(token_ids)
        logits = output.logits.log_softmax(dim=-1)
        return logits[..., self.label_idx]

    def score_texts(self, texts):
        token_ids = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        # The attention mask allows the model to ignore the padding tokens
        # So we can get the same output as calling score_text in a loop :)
        return self.score_token_ids(token_ids.input_ids, attention_mask=token_ids.attention_mask)

    def score_token_ids(self, token_ids: torch.Tensor, attention_mask=None):
        seq_len = token_ids.size(1)
        if seq_len > MAX_ALLOWED_SEQ_LEN:
            print(f"Warning: Sequence length exceeds maximum allowed length. Truncating to {MAX_ALLOWED_SEQ_LEN} tokens.")
            token_ids = token_ids[:, :MAX_ALLOWED_SEQ_LEN]
        output = self.model(token_ids, attention_mask=attention_mask)
        logits = output.logits.log_softmax(dim=-1)
        return logits[..., self.label_idx]
