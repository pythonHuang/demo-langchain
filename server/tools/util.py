from langchain_core.tools import tool

import os

@tool
def multiply(first_int: int, second_int: int) -> int:
    """两个整数相乘"""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers."""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Exponentiate the base to the exponent power."""
    return base**exponent

@tool
def list_files_in_directory(path: str) -> str:
    """List all file names in the directory"""
    file_names = os.listdir(path)

    # Join the file names into a single string, separated by a newline
    return "\n".join(file_names)
