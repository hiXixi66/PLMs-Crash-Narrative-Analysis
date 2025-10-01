import os
import json
import time
import hashlib
import requests
from typing import List, Dict, Union
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest



class LLM_HPC:

    def __init__(self, model_name: str = "llama3.2:1b", provider: str = "ollama", max_new_tokens = 1, cache_dir: str = "cache", model_dir: str = None):
        """
        Initialize LLM processing class
        :param model_name: LLM model to use
        :param provider: Select LLM service provider ('openai', 'ollama', 'transformers')
        :param cache_dir: Cache directory
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.model_dir = model_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if self.provider == "mistral" :
            mistral_models_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Mistral"
            self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
            self.model = Transformer.from_folder(mistral_models_path)

            # self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model")
            
        if self.provider == "bert":
            from transformers import BertTokenizer, BertForSequenceClassification
            if model_dir is None:
                model_dir = "models/bert-ft-MANCOLL-bias-test/with-bias-20perc-new/checkpoint-940"
            else:
                model_dir = self.model_dir
            # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/mancoll_bert2/checkpoint-845"
            self.tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = BertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            
            
        if self.provider == "transformers":
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if self.model_name == "qwen2.5-7b-instruct-1m":
                print("Loading Qwen2.5-7b-instruct-1m model...")
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/qwen2.5-7b-instruct-1m"
                # model_dir="/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/qwen2.5-finetune-crashtypeqkv/checkpoint-2163"
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/qwen-finetune-crashtypeq/checkpoint-1442"
                # model_dir = '/mimer/NOBACKUP/groups/naiss2025-22-321/projects/LLM-crash-data/models/qwen2.5-finetune-crashcat/checkpoint-72
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/projects/LLM-crash-data/models/qwen2.5-finetune-crashtype/checkpoint-2160"
                model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/qwen2.5-ft-MANCOLL/checkpoint-1251"
                use_auth_token="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"
                # assert os.path.exists(model_dir), f"路径不存在：{model_dir}"
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,local_files_only=True,use_auth_token=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
                
            elif self.model_name == "llama3-70b":
                if self.model_dir is not None:
                    model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/llama3/llama3-70b"
                else:
                    model_dir = self.model_dir
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
            elif self.model_name == "llama3-3b":
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/llama3-3b"
                # model_dir = "models/llama3B-ft-MANCOLL-noise-test/with-noise-40perc/checkpoint-1668"
                if self.model_dir is not None:
                    model_dir = self.model_dir
                else:
                    model_dir="models/llama3B-ft-MANCOLL-noise-test/with-bias-20perc/checkpoint-1668"
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama3-1b-finetune-crashtypeqkv/checkpoint-2884"
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama31b-ft-MANCOLL/checkpoint-1668"
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama3-3b-finetune-MANCOLL/checkpoint-1251"
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
            elif self.model_name == "llama3-8b":
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/llama3-8b"
                # model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama3-8b-finetune-crashtypeqkv/checkpoint-2163"
                model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/llama38b-ft-MANCOLL/checkpoint-1251"
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
            elif self.model_name == "deepseek-r1-Distill-Qwen-32B":
                model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
                model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/deepseek/deepseek-r1/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746"
                self.tokenizer = AutoTokenizer.from_pretrained( model_dir,trust_remote_code=True,local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype="auto"
                )
            elif self.model_name == "mistral" :
                # mistral_models_path = "/mimer/NOBACKUP/groups/naiss2025-22-321/Mistral"
                # self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
                # self.model = Transformer.from_folder(mistral_models_path)
                model_dir = "/mimer/NOBACKUP/groups/naiss2025-22-321/Cluster-LLM-Crash-Data/projects/LLM-crash-data/models/mistral-ft-MANCOLL/checkpoint-1251"
                # self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,local_files_only=True,use_auth_token=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )


    def _generate_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _load_cache(self, key: str) -> Union[str, None]:
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f).get("response")
        return None

    def _save_cache(self, key: str, response: str):
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False)

    def _query_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate text"""
        client = openai.OpenAI(api_key='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')
        
        max_tokens = self.max_new_tokens
        model = self.model_name
        # model = "gpt-3.5-turbo"
        # model = "llama-31-70b"
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        # print(response)
        return response.choices[0].message.content
    
    def _query_llama31_70b(self, prompt: str) -> str:
        """Call local Ollama to generate text"""
        client = openai.OpenAI(
            api_key="abc", base_url="http://llama-31-70b-service.ray:8000/v1/", default_headers={"x-foo": "true"}
        )
        max_tokens = self.max_new_tokens
        model="Meta-Llama-3.1-70B-Instruct"

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content
    
    
    def _query_transformers(self, prompt: str) -> str:
        """Call local Hugging Face Transformers to generate text"""
        # print(f"Loading model: {self.model_name}")
        
        input_text = prompt
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,pad_token_id=self.tokenizer.eos_token_id)
        # output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,temperature=0.2)
        

        # 只保留新生成部分（不包含输入 prompt）
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer
    
    def _query_mistral(self, prompt: str) -> str:
        """Call Mistral model to generate text"""

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], self.model, max_tokens=self.max_new_tokens, temperature=0.2, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result

    def batch_query(self, prompts: List[str], use_cache: bool = True) -> List[str]:
        """
        Batch query multiple prompt words
        :param prompts: prompt word list
        :param use_cache: whether to use cache
        :return: generate text list
        """
        return [self.query(prompt, use_cache=use_cache) for prompt in prompts]
    
    
    def _query_bert(self, prompt: str) -> str:
        """Call local BERT model to generate text"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        return str(predicted_class_id)
    
    def query(self, prompt: str):
        provider = self.provider
        if provider == "openai":
            return self._query_openai(prompt)
        elif provider == "transformers":
            return self._query_transformers(prompt)
        elif provider == "llama31":
            return self._query_llama31_70b(prompt)
        elif provider == "mistral":
            return self._query_mistral(prompt)
        elif provider == "bert":
            return self._query_bert(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")




# if __name__ == "__main__":
#     llm = LLM()


#     response = llm.query("What is the capital of France?")
#     print("OpenAI GPT-4 Response:", response)


#     responses = llm.batch_query(["Tell me a joke.", "What is the meaning of life?"])
#     print("Batch Responses:", responses)
