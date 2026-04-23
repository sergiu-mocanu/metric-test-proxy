import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from test_proxy.codegen.interface import CodeGenerator, GenerationConfig


class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)


class TransformersGenerator(CodeGenerator):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # noqa
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=0
        )

        self.stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.tokenizer, ['\n\ndef ', '\nclass ', '\n#'])
        ])


    def generate(self, prompt: str, config: GenerationConfig):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            # stopping_criteria=self.stopping_criteria
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text