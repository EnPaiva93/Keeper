from transformers import PretrainedConfig
from typing import List


class KeeperConfig(PretrainedConfig):
    model_type = "keeper"

    def __init__(
        self,
        retriever_config = {
          "_name_or_path": "AdrienB134/ColBERTv1.0-bert-based-spanish-mmarcoES",
          "architectures": [
            "HF_ColBERT"
          ],
          "attention_probs_dropout_prob": 0.1,
          "classifier_dropout": None,
          "gradient_checkpointing": False,
          "hidden_act": "gelu",
          "hidden_dropout_prob": 0.1,
          "hidden_size": 768,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "layer_norm_eps": 1e-12,
          "max_position_embeddings": 512,
          "model_type": "bert",
          "num_attention_heads": 12,
          "num_hidden_layers": 12,
          "output_past": True,
          "pad_token_id": 1,
          "position_embedding_type": "absolute",
          "torch_dtype": "float32",
          "transformers_version": "4.35.2",
          "type_vocab_size": 2,
          "use_cache": True,
          "vocab_size": 31002
        },
        model_config = {
          "_name_or_path": "google/gemma-2b-it",
          "architectures": [
            "GemmaForCausalLM"
          ],
          "attention_bias": False,
          "attention_dropout": 0.0,
          "bos_token_id": 2,
          "eos_token_id": 1,
          "head_dim": 256,
          "hidden_act": "gelu",
          "hidden_size": 2048,
          "initializer_range": 0.02,
          "intermediate_size": 16384,
          "max_position_embeddings": 8192,
          "model_type": "gemma",
          "num_attention_heads": 8,
          "num_hidden_layers": 18,
          "num_key_value_heads": 1,
          "pad_token_id": 0,
          "rms_norm_eps": 1e-06,
          "rope_scaling": None,
          "rope_theta": 10000.0,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.38.0.dev0",
          "use_cache": True,
          "vocab_size": 256000
        },
        auto_map = {     
          "AutoConfig": "configuration_keeper.KeeperConfig",     
          "AutoModel": "tokenizer_keeper.KeeperTokenizer",
          "AutoModelForCausalLM": "model_keeper.KeeperModelForCausalLM",  
        },
        **kwargs,
    ):
        self.retriever_config = retriever_config
        self.model_config = model_config
        self.device_map = 'auto'
        self.auto_map = auto_map
        
        super().__init__(**kwargs)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
