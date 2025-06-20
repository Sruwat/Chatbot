from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=256, stream=False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        if stream:
            for sentence in result.split('. '):
                yield sentence.strip() + '. '
        else:
            return result

def generate_answer(context, question, stream=False):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    llm = LocalLLM()
    if stream:
        return llm.generate(prompt, stream=True)
    else:
        return llm.generate(prompt)

if __name__ == "__main__":
    context = "This is a sample context from the document."
    question = "What is this context about?"
    print(generate_answer(context, question))