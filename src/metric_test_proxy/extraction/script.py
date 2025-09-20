import re


def tokenize(raw_string: str) -> list[str]:
    """Tokenize input code into a list of tokens."""
    return re.findall(r"\w+|[^\w\s]", raw_string)


def extract_script(script: str, remove_assert: bool=False, remove_exit: bool=False) -> str:
    """Extract the code from a raw string. Remove all comments in order to avoid unnecessary textual-similarity noise.

    Args:
        script (str): raw string of the script
        remove_assert (bool): remove any assert statements
        remove_exit (bool): remove any exit statements

    Returns:
        str: extracted code
    """
    if 'METADATA' in script:
        script = script.split('METADATA', 1)[0]
    elif 'def check(candidate)' in script:
        script = script.split('def check(candidate)', 1)[0]

    script_lines = script.splitlines()

    multi_line_comment = False
    comment_index = []
    assert_index = []
    empty_line_index = []
    exit_line_index = []

    for index, line in enumerate(script_lines):

        # Index all assert statements
        if remove_assert and 'assert' in line:
            line_elements = tokenize(line)
            if line_elements[0] == 'assert':
                assert_index.append(index)
            continue

        if remove_exit and 'exit(' in line:
            exit_line_index.append(index)
            continue

        if not multi_line_comment:
            if '#' in line:
                # Index single-line comments
                if line.strip()[0] == '#':
                    comment_index.append(index)
                # Remove in-line comment component
                else:
                    cleaned_up_line = line.split('#', 1)[0]
                    script_lines[index] = cleaned_up_line
                continue

            # Index the first line of multi-line comments
            if '"""' in line or "'''" in line:
                comment_index.append(index)
                if line.count('"""') == 1 or line.count("'''") == 1:
                    multi_line_comment = True
                continue

        # Add indexes for multi-line comments
        if multi_line_comment and ('"""' not in line and "'''" not in line):
            comment_index.append(index)
            continue

        # Index the last line of multi-line comments
        if multi_line_comment and ('"""' in line or "'''" in line):
            multi_line_comment = False
            comment_index.append(index)
            continue

        # Index blank lines
        if len(line) == 0 or line.isspace():
            empty_line_index.append(index)
            continue

    # Merge indexes for comments, empty lines, assert and exit statements
    [comment_index.extend(indexes) for indexes in (empty_line_index, assert_index, exit_line_index)]

    # Remove all the unnecessary script components
    for index in sorted(comment_index, reverse=True):
        del script_lines[index]

    clean_script = '\n'.join(script_lines)
    return clean_script


def extract_tests(humaneval_script: str) -> str:
    """Extract test suite from HumanEval ground truth implementation"""
    testing_funct_name = 'def check('

    extracted_checker = humaneval_script.split(testing_funct_name, 1)[1]
    test_funct = testing_funct_name + extracted_checker

    list_lines = test_funct.split('\n')

    del_index = []

    # Index empty lines, comments and unnecessary assert statements
    for index, line in enumerate(list_lines):
        if (len(line) == 0
                or line.isspace()
                or line.strip().startswith('#')
                or 'assert True' in line):
            del_index.append(index)

    # Remove unnecessary components
    for index in reversed(del_index):
        del list_lines[index]

    test_funct = '\n'.join(list_lines)
    return test_funct
