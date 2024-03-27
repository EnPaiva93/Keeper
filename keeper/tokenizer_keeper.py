import os
import json

from keeper.configuration_keeper import KeeperConfig
from transformers import PreTrainedTokenizer
from typing import Optional, Union


class KeeperTokenizer(PreTrainedTokenizer):

    config_class = KeeperConfig

    def __init__(self, cfg=None):


        self.tokenizer_retriever = None
        self.tokenizer_model = None

        if cfg:
            print("Initializing KeeperTokenizer with cfg")
            # Inicialización con configuración
            self.tokenizer_retriever = AutoTokenizer.from_pretrained(cfg.retriever_config['_name_or_path'])
            self.tokenizer_model = AutoTokenizer.from_pretrained(cfg.model_config['_name_or_path'])

            # Almacena kwargs para la serialización y carga futura
            self.init_kwargs = {'cfg': cfg}

            super().__init__()  # Inicializa la clase base al principio
            print("Initialization complete")
        else:
            # Si cfg no se proporciona, esto se manejará en el método from_pretrained
            print("Initializing KeeperTokenizer without cfg")



    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Crea una nueva instancia de KeeperTokenizer sin cfg
        instance = cls()

        print("Loading tokenizer_retriever from", pretrained_model_name_or_path)
        instance.tokenizer_retriever = AutoTokenizer.from_pretrained(
            os.path.join(pretrained_model_name_or_path, 'tokenizer-retriever')
        )

        print("Loading tokenizer_model from", pretrained_model_name_or_path)
        instance.tokenizer_model = AutoTokenizer.from_pretrained(
            os.path.join(pretrained_model_name_or_path, 'tokenizer-model')
        )

        return instance

    @property
    def vocab_size(self):
        # Obtiene los vocabularios de ambos tokenizadores
        vocab_retriever = self.tokenizer_retriever.get_vocab()
        vocab_model = self.tokenizer_model.get_vocab()
        
        # Combina los vocabularios
        combined_vocab = {**vocab_retriever, **vocab_model}
        
        # Devuelve el tamaño del vocabulario combinado
        return len(combined_vocab)


    def get_vocab(self):
        # Obtiene los vocabularios de ambos tokenizadores
        vocab_retriever = self.tokenizer_retriever.get_vocab()
        vocab_model = self.tokenizer_model.get_vocab()

        # Organiza los vocabularios en un diccionario con claves separadas
        separated_vocabularies = {
            'vocab_retriever': vocab_retriever,
            'vocab_model': vocab_model
        }

        return separated_vocabularies

    def _tokenize(self, text, **kwargs):
        # You must implement this method for your tokenization logic
        pass

    def encode(self, text, **kwargs):
        tokens_retriever = self.tokenizer_retriever(text, return_tensors='pt', **kwargs)
        tokens_model = self.tokenizer_model(text, return_tensors='pt', **kwargs)

        return {
            'tokens_retriever': tokens_retriever,
            'tokens_model': tokens_model
        }

    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
      return self.tokenizer_model.decode(token_ids, skip_special_tokens, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        # Asegúrate de que el directorio de salida existe
        os.makedirs(save_directory, exist_ok=True)

        # Guarda el tokenizador retriever
        retriever_save_directory = os.path.join(save_directory, "tokenizer-retriever")
        os.makedirs(retriever_save_directory, exist_ok=True)
        self.tokenizer_retriever.save_pretrained(retriever_save_directory)

        # Guarda el tokenizador model
        model_save_directory = os.path.join(save_directory, "tokenizer-model")
        os.makedirs(model_save_directory, exist_ok=True)
        self.tokenizer_model.save_pretrained(model_save_directory)
      
        # Devuelve los nombres de los archivos guardados (opcional)
        saved_files = [
            "tokenizer-retriver/tokenizer_config.json",
            "tokenizer-retriver/special_tokens_map.json",
            "tokenizer-retriver/vocab.json",
            "tokenizer-retriver/added_tokens.json",
            "tokenizer-model/tokenizer_config.json",
            "tokenizer-model/special_tokens_map.json",
            "tokenizer-model/vocab.json",
            "tokenizer-model/added_tokens.json"
        ]
        return tuple(os.path.join(save_directory, file) for file in saved_files)
