import torch
from angle_emb import AnglE, Prompts


class LlamaSenetenceEmbed():
    def __init__(self, device='cuda:0'):
        print(device)
        self.angle = AnglE.from_pretrained(
            'NousResearch/Llama-2-7b-hf',
            pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
            pooling_strategy='last',
            is_llm=True,
            torch_dtype=torch.float16,
            device=device)
        self.prompt = Prompts.A

    def encode(self, text):
        texts = [{'text': i} for i in text]
        doc_vecs = self.angle.encode(texts, prompt=Prompts.A, to_numpy=False)
        return doc_vecs

    def save_pretrained(self, path):
        self.angle.save_pretrained(path)


if __name__ == "__main__":
    llama = LlamaSenetenceEmbed()
    emb = llama.encode(['hello world', 'goodbye world'])
    print(emb)
