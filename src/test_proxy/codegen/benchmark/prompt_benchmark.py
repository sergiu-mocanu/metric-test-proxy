import os
from dataclasses import dataclass

from test_proxy.codegen.interface import GenerationConfig
from test_proxy.codegen.backends.transformers_backend import TransformersGenerator
from test_proxy.codegen.prompts import humaneval_prompt
from test_proxy.extraction.script_components import get_function_metadata

config = GenerationConfig()


@dataclass
class Task:
    id: int
    signature: str
    docstring: str


list_tasks: list[Task] = []


def populate_list_tasks():
    humaneval_path = '/home/zara/dev/metric_exp_data/code/humaneval_baseline'
    for index, file in enumerate(sorted(os.listdir(humaneval_path))):
        with open(os.path.join(humaneval_path, file), 'r') as f:
            content = f.read()

        signature, docstring = get_function_metadata(content)
        task = Task(
            id=index,
            signature=signature,
            docstring=docstring
        )
        list_tasks.append(task)


def evaluate_prompt(generator, prompt_fn):
    results = []

    populate_list_tasks()

    for task in list_tasks:
        prompt = prompt_fn(task.signature, task.docstring)
        raw = generator.generate(prompt, config)

        res_entry = {
            'task_id': task.id,
            'raw': raw,
        }

        print(f'Task id: {task.id}\n')
        print(raw)
        print('_' * 60)

        results.append(res_entry)

    return results

current_generator = TransformersGenerator('bigcode/starcoder2-7b')

benchmark_results = evaluate_prompt(current_generator, humaneval_prompt)
