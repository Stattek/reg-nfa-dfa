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


class PostfixRegex:
    """Contains methods to convert a regular expression from infix to postfix and evaluate it."""

    def __init__(self, infix_regex):
        """Creates a PostfixRegex from a regular expression in infix.

        Args:
            regex (str): The infix regular expression.
        """
        # represent a regular expression as a stack
        self._postfix_regex = self.__infix_to_postfix(infix_regex)

    def generate_nfa(self) -> NFA:
        """Evaluates and parses the regex into an NFA.

        Returns:
            NFA: The output NFA.
        """
        pass

    def __get_precedence(self, operator):
        if operator == "*":
            return Operator.STAR_CLOSURE.value
        if operator == ".":
            return Operator.CONCATENATION.value
        # assume that we have alternation if nothing else matches
        # (the string shouldn't have any bad characters, since we already validated it)
        return Operator.ALTERNATION.value

    def __infix_to_postfix(self, regex_str: str) -> str:
        """Converts the regular expression string to postfix from infix.

        Args:
            regex_str (str): The regular expression string.

        Returns:
            str: The output postfix string.
        """
        operators = "*.|"
        output = ""
        stack = []
        for char in regex_str:
            if char not in operators:
                # if we have an operand
                output += char
            else:
                # if we have an operator

                # if the stack is empty or the precedence of the current character
                # is greater than that of the top of the stack or there is a
                # parenthesis in the stack
                if (
                    len(stack) == 0
                    or self.__get_precedence(char) > self.__get_precedence(stack[0])
                    or "(" in stack
                ):
                    # push this operator onto the stack
                    stack.insert(0, char)
                else:
                    # pop operators from stack until we find one
                    # with a lower precedence (or an empy stack)
                    while len(stack) != 0 and self.__get_precedence(
                        char
                    ) <= self.__get_precedence(stack[0]):
                        # add popped values to output
                        output += stack.pop()
                    # now we can push this operator to the stack that
                    stack.insert(0, char)

                # check for parentheses
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
    """Contains methods to validate a regular expression."""

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
            # DEBUG: remove the print below
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

    postfix_regex = PostfixRegex(regex_str)
    # DEBUG: remove the print below
    print("final output=", postfix_regex._postfix_regex)


if __name__ == "__main__":
    main()
