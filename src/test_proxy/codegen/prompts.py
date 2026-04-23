def humaneval_prompt(signature: str, docstring: str) -> str:
    return f"""You are a Python expert.

Write a correct implementation for the following function.

{signature}
{docstring}

Return ONLY the Python function implementation. No explanation.
Stop immediately after the function body ends."""