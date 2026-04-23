from test_proxy.extraction.script_components import extract_code
from test_proxy.codegen.interface import GenerationConfig

config = GenerationConfig()

def evaluate_prompt(generator, prompt_fn, tasks):
    results = []

    for task in tasks:
        prompt = prompt_fn(task.signature, task.docstring)
        raw = generator.generate(prompt, config)
        code = extract_code(raw)

        results.append({
            'task_id': task.id,
            'raw': raw,
            'code': code,
        })

    return results