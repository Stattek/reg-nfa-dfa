#!/usr/bin/env python3
import sys


# class to represent an NFA
class NFA:
    pass


# class to validate regular expressions
class RegexValidator:

    def validate_regex(regex_str: str) -> bool:
        """Validates a regular expression.

        Args:
            regex_str (str): The regular expression string.

        Returns:
            bool: True if the string is valid, False otherwise.
        """
        # check if the number of parentheses is consistent
        if regex_str.count("(") != regex_str.count(")"):
            return False

        # remove empty parentheses
        regex_str = RegexValidator.__remove_empty_parentheses(regex_str)

        return True

    def __remove_empty_parentheses(regex_str: str) -> str:
        """Removes empty parentheses from a string

        Args:
            regex_str (str): The regular expression string.

        Returns:
        str: The modified string.
        """
        empty_parentheses = "()"
        # iteratively remove all empty sets of parentheses
        regex_idx = regex_str.find(empty_parentheses)
        while regex_idx >= 0:
            # remove this set of parentheses
            regex_str = regex_str[:regex_idx] + regex_str[regex_idx + 2 :]
            regex_idx = regex_str.find(empty_parentheses)

        return regex_str


def main():
    # take arguments
    if len(sys.argv) != 3:
        # invalid number of arguments
        print("Usage: {} <regex> <file_name>".format(sys.argv[0]))
        sys.exit(1)
    regex_str = sys.argv[1]

    # validate regular expression
    if not RegexValidator.validate_regex(regex_str):
        # invalid regular expression
        print("{} is not a valid regular expression".format(regex_str))
        sys.exit(1)


if __name__ == "__main__":
    main()
