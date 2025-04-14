#!/usr/bin/env python3
import sys


def validate_regex(regex_str: str):
    """Validates a regular expression.

    Args:
        regex_str (str): The regular expression string.

    Returns:
        bool: True if the string is valid, False otherwise.
    """
    # check if the number of parentheses is consistent
    if regex_str.count("(") != regex_str.count(")"):
        return False
    return True


def main():
    # take arguments
    if len(sys.argv) != 3:
        # invalid number of arguments
        print("Usage: {} <regex> <file_name>".format(sys.argv[0]))
    regex_str = sys.argv[1]

    # validate regular expression
    if not validate_regex(regex_str):
        # invalid regular expression
        print("{} is not a valid regular expression".format(regex_str))
        sys.exit(1)


if __name__ == "__main__":
    main()
