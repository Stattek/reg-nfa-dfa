#!/usr/bin/env python3
import sys
from enum import Enum


# class to represent an NFA
class NFA:
    pass


class Operator(Enum):
    """enum to represent an operator."""

    ALTERNATION = 0
    CONCATENATION = 1
    STAR_CLOSURE = 2
    UNKNOWN = 3


class Regex:
    def __init__(self, regex):
        # represent a regular expression as a stack
        self.regex_queue = self.create_regex_from_str(regex)

    def __parse_regex(regex: str):
        # base case
        if regex == "":
            return

    def create_regex_from_str(regex: str) -> list:
        output = []


class InfixToPostfix:
    def get_precedence(operator):
        if operator == "*":
            return Operator.STAR_CLOSURE.value
        elif operator == ".":
            return Operator.CONCATENATION.value
        else:
            return Operator.ALTERNATION.value

    def infix_to_postfix(regex_str: str) -> str:
        operators = "*.|"
        output = ""
        stack = []
        for char in regex_str:
            if char not in operators:
                # operand
                output += char
            else:
                if (
                    len(stack) == 0
                    or InfixToPostfix.get_precedence(char)
                    > InfixToPostfix.get_precedence(stack[0])
                    or "(" in stack
                ):
                    # push this operator onto the stack
                    stack.insert(0, char)
                else:
                    # pop operators from stack until we find one with a lower precedence (or an empy stack)
                    while len(stack) != 0 and InfixToPostfix.get_precedence(
                        char
                    ) <= InfixToPostfix.get_precedence(stack[0]):
                        output += stack.pop()
                    # push this operator to the stack
                    stack.insert(0, char)

                if char == "(":
                    # push to stack
                    stack.insert(0, char)
                elif char == ")":
                    # pop from the stack until we find the matching parenthesis
                    while stack[0] != "(":
                        output += stack.pop()
        # add remaining values on the stack
        while len(stack) > 0:
            output += stack.pop()

        return output


# class to validate regular expressions
class RegexValidator:
    def insert_alternation_operator(regex_str: str) -> str:
        """Inserts the alternation operator in a regular expression. For internal use.
        This is to make conversion to postfix easier.

        Args:
            regex_str (str): The input regular expression.

        Returns:
            str: The modified regular expression.
        """
        import re

        pattern_str = r"(?<=[a-zA-Z])(?=[a-zA-Z])|(?<=\))(?=\()|(?<=[a-zA-Z])(?=\()|(?<=\))(?=[a-zA-Z])"
        pattern = re.compile(pattern_str)

        match = pattern.search(regex_str)
        while match:
            regex_str = regex_str[: match.start()] + "." + regex_str[match.end() :]
            print(regex_str)
            match = pattern.search(regex_str)

        return regex_str

    def validate_and_modify_regex(regex_str: str) -> tuple[bool, str]:
        """Validates a regular expression and modifies it for use with
        other functions.

        Args:
            regex_str (str): The regular expression string.

        Returns:
            tuple[bool, str]: True if the string is valid, with the regex result string.
            False otherwise, with None.
        """
        # check if the number of parentheses is consistent
        if regex_str.count("(") != regex_str.count(")"):
            return (False, None)

        # check that the regular expression only has accepted characters
        accepted_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()*|"
        for char in regex_str:
            if char not in accepted_chars:
                return (False, None)

        regex_str = RegexValidator.insert_alternation_operator(regex_str)

        return (True, regex_str)


def main():
    # take arguments
    if len(sys.argv) != 3:
        # invalid number of arguments
        print("Usage: {} <regex> <file_name>".format(sys.argv[0]))
        sys.exit(1)
    regex_str = sys.argv[1]

    # validate regular expression
    (is_valid, regex_str) = RegexValidator.validate_and_modify_regex(regex_str)
    if not is_valid:
        # invalid regular expression
        print("{} is not a valid regular expression".format(regex_str))
        sys.exit(1)

    regex_str = InfixToPostfix.infix_to_postfix(regex_str)
    print("final output=", regex_str)


if __name__ == "__main__":
    main()
