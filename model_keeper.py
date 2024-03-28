from .configuration_keeper import KeeperConfig

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    PreTrainedModel, 
    PretrainedConfig,
    AutoModelForCausalLM
)

from typing import Dict
import torch
import numpy as np
from einops import rearrange

class KeeperModelForCausalLM(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """
    config_class = KeeperConfig
    base_model_prefix = "keeper_model"

    def __init__(self, cfg, n_cands=8, update_both=False) -> None:
        super().__init__(cfg)

        self.bert = None
        self.llm = None

        if cfg:
            print("Initializing KeeperModelForCausalLM from cfg")
            # Inicialización con configuración

            self.bert = AutoModel.from_pretrained(cfg.retriever_config['_name_or_path'])
            self.llm = AutoModelForCausalLM.from_pretrained(
                cfg.model_config['_name_or_path'],
                device_map=cfg.device_map
                )

            # Almacena kwargs para la serialización y carga futura
            # self.init_kwargs = {'cfg': cfg}

            print("Initialization complete")
        else:
            # Si cfg no se proporciona, esto se manejará en el método from_pretrained
            print("Initializing KeeperTokenizer without cfg")

        self.n_cands = n_cands
        self.update_both = update_both
        print(f"Model n_cands: {self.n_cands}")

        # Inicializar buffers vacíos para document_vecs y document_mask
        self.register_buffer('document_retriever_text', torch.empty(0, dtype=torch.long))
        self.register_buffer('document_retriever_mask', torch.empty(0, dtype=torch.long))
        self.register_buffer('document_retriever_type', torch.empty(0, dtype=torch.long))
        self.register_buffer('document_model_text', torch.empty(0, dtype=torch.long))
        # self.register_buffer('document_model_mask', torch.empty(0, dtype=torch.long))
        # self.register_buffer('document_model_type', torch.empty(0, dtype=torch.long))
        self.register_buffer('prompt_left', torch.empty(0, dtype=torch.long))
        self.register_buffer('prompt_right', torch.empty(0, dtype=torch.long))
        self.register_buffer('respuesta', torch.empty(0, dtype=torch.long))

    def generate(self, query: Dict[str, torch.LongTensor], k: int = 3, **kwargs):

        query_retriever = {k: v.to("cuda") for k, v in query['tokens_retriever'].items()}
        query_model = {k: v.to("cuda") for k, v in query['tokens_model'].items()}

        query_vecs = self.forward_representation(query_retriever)

        doc_dic = {'input_ids': self.document_retriever_text, 'attention_mask':self.document_retriever_mask, 'token_type_ids': self.document_retriever_type}

        document_vecs = self.forward_representation(doc_dic, sequence_type="doc")

        self.score = self.forward_aggregation(query_vecs, query['tokens_model']["attention_mask"], document_vecs, self.document_retriever_mask)

        k = min(k, self.score.numel())

        topk_scores, topk_indices = torch.topk(self.score, k)

        topk_texts = [self.document_model_text[i] for i in topk_indices[0].tolist()]

        concatenated_texts = torch.cat(topk_texts, dim=0)

        T = torch.cat((self.prompt_left, concatenated_texts.unsqueeze(0), self.prompt_right, query_model['input_ids'], self.respuesta), dim=1)

        outputs = self.llm.generate(input_ids=T, max_new_tokens=256, repetition_penalty=1.15)

        return outputs

    def forward_representation(self,
                               tokens,
                               max_seq_len = 512,
                               sequence_type=None) -> torch.Tensor:

        if sequence_type == "doc":
            if self.update_both:
              with torch.no_grad():
                vecs = self.bert(**tokens)[0]
            else:
              with torch.no_grad():
                with torch.no_grad():
                    vecs = self.bert(**tokens)[0] # assuming a distilbert model here
        else:
          with torch.no_grad():
            vecs = self.bert(**tokens)[0]
        # vecs = self.compressor(vecs)
        return vecs

    def forward_aggregation(self, query_vecs, query_mask, document_vecs, document_mask):

        # query_vecs: B x N x D
        # doc_vecs: (B * k) x N x D

        # Unsqueeze query vector
        _bsz = query_vecs.shape[0]
        n_cands = document_vecs.shape[0] // _bsz
        query_vecs_dup = query_vecs.repeat_interleave(n_cands, dim=0).contiguous()

        score = torch.bmm(query_vecs_dup, document_vecs.transpose(1, 2))
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values
        query_mask_dup = query_mask.repeat_interleave(n_cands, dim=0).contiguous()

        score[~(query_mask_dup.bool())] = 0
        score = rearrange(score.sum(-1), '(b n) -> b n', n=n_cands) # B x k
        return score

    def prompt(self, left_p = None, right_p = None):
        if left_p is None:
          left_p = """ Eres un experto en cultura paraguaya que responde segun el contexto:
-------------------------------
"""
        if right_p is None:
          right_p = """
-------------------------------
- Debes responder solamente en Espanol
- No utilices conocimientos previos.
- Responde de forma clara, amable y concisa.

Pregunta: """
        return left_p, right_p

    def save_docs(self, docs: list, tokenizer, max_seq_len=512):
        # Tokenizamos el prompt
        prompt_left, prompt_right = self.prompt()
        prompt_left_output = tokenizer.encode(prompt_left)
        prompt_right_output = tokenizer.encode(prompt_right)

        # Tokenizamos el documento
        doc_outputs = tokenizer.encode(docs, max_length=max_seq_len, padding='max_length', truncation=True)

        # Tokenizamos la Respuesta
        resp = tokenizer.encode('\nRespuesta: ')
        resp_model = {k: v.to("cuda") for k, v in resp['tokens_model'].items()}

        # Pasamos los tensores a cuda  (## optimizar: se guardan tensores que no se utilizaran en la gpu)
        doc_outputs = {k: v.to("cuda") for k, v in doc_outputs.items()}
        prompt_left_output = {k: v.to("cuda") for k, v in prompt_left_output.items()}
        prompt_right_output = {k: v.to("cuda") for k, v in prompt_right_output.items()}

        # Actualizar el buffer con los vectores de documentos
        self.document_retriever_text = doc_outputs['tokens_retriever']['input_ids']
        self.document_retriever_mask = doc_outputs['tokens_retriever']['attention_mask']
        self.document_retriever_type = doc_outputs['tokens_retriever']['token_type_ids']
        self.document_model_text = doc_outputs['tokens_model']['input_ids']
        # self.document_model_mask = key_outputs['tokens_model']['attention_mask']
        # self.document_model_type = key_outputs['tokens_model']['token_type_ids']
        self.prompt_left = prompt_left_output['tokens_model']['input_ids']
        self.prompt_right = prompt_right_output['tokens_model']['input_ids']
        self.respuesta = resp_model['tokens_model']['input_ids']
