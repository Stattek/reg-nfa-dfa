#!/usr/bin/env python3
import sys
from enum import Enum

OPERATORS = "*.|"
PARENTHESES = "()"


class Node:
    """Represents a Node in an NFA/DFA."""

    def __init__(self, transition_dict, is_accepting):
        """Create a new Node.

        Args:
            transition_dict (dict): Dictionary for transitions from this Node.
            is_accepting (bool): If this Node is accepting.
        """
        # keeps track of the alphabet (key) and holds a value
        # which is a list of indices to other nodes in the graph
        self.transition_dict = transition_dict
        self.is_accepting = is_accepting

    def to_str(self, sigma):
        """Converts this Node to string.

        Args:
            sigma (list): The alphabet for the NFA/DFA this Node belongs to.

        Returns:
            str: The string representation of this Node.
        """
        output = ""

        sigma.append("")
        for i, key in enumerate(sigma):
            if key == "":
                output += '""'
            else:
                output += key
            output += "-> {"

            try:
                for j, value in enumerate(self.transition_dict[key]):
                    output += str(value)
                    if j != len(self.transition_dict[key]) - 1:
                        output += ", "
            except:
                # don't print anything if there is an error
                pass
            output += "}"
            if i != len(sigma) - 1:
                output += " "
        sigma.pop()  # remove the empty string we added earlier

        return output


# class to represent an NFA
class NFA:
    """Represents an NFA. Able to create an NFA from regular expressions."""

    def __init__(self):
        """Creates a new empty NFA."""
        self._nodes = []
        self._initial_state = None
        self._accepting_states = []
        self._sigma = []

    def __str__(self):
        """Converts this NFA to string.

        Returns:
            str: The string representation of the NFA.
        """
        output = "NFA:\n"
        output += "Sigma: "
        for i, val in enumerate(self._sigma):
            output += str(val)
            if i != len(self._sigma) - 1:
                output += " "
        output += "\n"
        output += "------\n"

        str(self._sigma) + "\n"
        # print nodes
        for i, node in enumerate(self._nodes):
            output += str(i) + ": " + node.to_str(self._sigma) + "\n"
        output += "------\n"
        output += str(self._initial_state) + ": Initial State\n"

        # print accepting states
        for i, value in enumerate(self._accepting_states):
            output += str(value)
            if i != len(self._accepting_states) - 1:
                output += ","
        output += ": Accepting State(s)"
        return output

    def __combine_sigma(self, lhs, rhs):
        """Combines the sigmas of two NFAs.

        Args:
            lhs (NFA): First NFA to combine.
            rhs (NFA): Second NFA to combine.

        Returns:
            list: The new sigma from combining unique values of the lhs and rhs.
        """
        output = []
        for value in lhs:
            if value not in output:
                output += value
        for value in rhs:
            if value not in output:
                output += value
        return output

    def create_single_char_nfa(char: chr):
        """Creates an NFA to accept a single character.

        Args:
            char (chr): The character to accept.

        Returns:
            NFA: The NFA to accept a single character.
        """
        # simplest NFA is just an initial (nonaccepting) state
        # that transitions on the character to an accepting state
        nfa = NFA()
        nfa._nodes = [Node({char: [1]}, False), Node({}, True)]
        nfa._initial_state = 0
        nfa._accepting_states = [1]
        nfa._sigma = [char]
        return nfa

    def __convert_to_nfa(self, maybe_nfa):
        """Converts the input parameter to an NFA.

        Args:
            maybe_nfa (NFA or str): Can either be an NFA to then\
            return, or a single character string to convert to NFA.

        Returns:
            NFA: The NFA if the value was already an NFA\
            or a newly created NFA from the input string.
        """
        if isinstance(maybe_nfa, str):
            # create an NFA out of this symbol
            return NFA.create_single_char_nfa(maybe_nfa)
        else:
            # assume this is already an NFA
            return maybe_nfa

    def __displace_transition_dict(self, nfa, amount):
        """Displaces a transition dictionary of an NFA by a fixed value.

        Args:
            nfa (NFA): The NFA to displace the transition dictionary of.
            amount (int): The amount to displace the transition dictionary by.

        Returns:
            NFA: The NFA with its transition indices displaced by the specified amount.
        """
        new_nfa = NFA()
        new_nfa._initial_state = nfa._initial_state
        new_nfa._accepting_states = nfa._accepting_states
        new_nfa._sigma = nfa._sigma

        nodes = []
        for node in nfa._nodes:
            # make a copy of this node
            new_node_is_accepting = node.is_accepting
            new_node_transition_dict = {}

            for transition, values in node.transition_dict.items():
                new_node_transition_dict[transition] = [
                    value + amount for value in values
                ]

            new_node = Node(new_node_transition_dict, new_node_is_accepting)
            nodes.append(new_node)

        new_nfa._nodes = nodes
        return new_nfa

    def __alternation(self, lhs, rhs):
        """Performs alternation on the two NFAs.

        Args:
            lhs (NFA): The left-hand side NFA.
            rhs (NFA): The right-hand side NFA.

        Returns:
            NFA: The result of alternation.
        """
        lhs = self.__convert_to_nfa(lhs)
        rhs = self.__convert_to_nfa(rhs)

        # setup the new start NFA
        new_start = NFA()
        # we will have lambda transitions to the start of the lhs and rhs
        new_start._nodes = [
            Node(
                {
                    "": [
                        lhs._initial_state + 1,
                        rhs._initial_state + 1 + len(lhs._nodes),
                    ]
                },
                False,
            )
        ]
        new_start._initial_state = 0
        new_start._accepting_states = []

        # save the initial nodes for later
        new_start_inital_num_nodes = len(new_start._nodes)

        # setup lhs and rhs
        lhs = self.__convert_to_nfa(lhs)
        rhs = self.__convert_to_nfa(rhs)
        lhs = self.__displace_transition_dict(lhs, new_start_inital_num_nodes)
        rhs = self.__displace_transition_dict(
            rhs, len(lhs._nodes) + new_start_inital_num_nodes
        )

        # add the nodes and accepting states nodes
        new_start._nodes.extend(lhs._nodes)
        new_start._nodes.extend(rhs._nodes)
        for val in lhs._accepting_states:
            val += new_start_inital_num_nodes
            new_start._accepting_states.append(val)
        for val in rhs._accepting_states:
            val += len(lhs._nodes) + new_start_inital_num_nodes
            new_start._accepting_states.append(val)

        # combine sigma
        new_sigma = self.__combine_sigma(lhs._sigma, rhs._sigma)
        new_sigma = self.__combine_sigma(new_sigma, new_start._sigma)
        new_start._sigma = new_sigma
        return new_start

    def __concatenation(self, lhs, rhs):
        """Performs concatenation on the two NFAs.

        Args:
            lhs (NFA): The left-hand side NFA.
            rhs (NFA): The right-hand side NFA.

        Returns:
            NFA: The result of concatenation.
        """
        lhs = self.__convert_to_nfa(lhs)
        rhs = self.__convert_to_nfa(rhs)
        lhs_inital_num_nodes = len(lhs._nodes)

        rhs = self.__displace_transition_dict(rhs, lhs_inital_num_nodes)
        for accepting_state in lhs._accepting_states:
            lhs._nodes[accepting_state].is_accepting = False
            # set the transition to the initial state of rhs (should always be 0)
            # also, since we are going to append the node "lists," we add the
            # length of the nodes list to the initial state index
            try:
                # try appending
                lhs._nodes[accepting_state].transition_dict[""].append(
                    len(lhs._nodes) + rhs._initial_state
                )
            except KeyError:
                # create a new list on error
                lhs._nodes[accepting_state].transition_dict[""] = [
                    len(lhs._nodes) + rhs._initial_state
                ]

        # add the nodes and accepting states nodes
        lhs._accepting_states = []  # since we turned all accepting states false
        lhs._nodes.extend(rhs._nodes)
        for val in rhs._accepting_states:
            val += lhs_inital_num_nodes
            lhs._accepting_states.append(val)

        # combine sigma
        new_sigma = self.__combine_sigma(lhs._sigma, rhs._sigma)
        lhs._sigma = new_sigma
        return lhs

    def __star_closure(self, operand):
        """Performs star closure on the two NFAs.

        Args:
            operand (NFA): The NFA to perform star closure on.

        Returns:
            NFA: The result of star closure.
        """
        new_start = NFA()
        # we will have a lambda transition to the start of the operand
        operand = self.__convert_to_nfa(operand)

        new_start._nodes = [Node({"": [operand._initial_state + 1]}, True)]
        new_start._initial_state = 0
        new_start._accepting_states = [0]
        new_start_inital_num_nodes = len(new_start._nodes)

        operand = self.__displace_transition_dict(operand, new_start_inital_num_nodes)

        for accepting_state in operand._accepting_states:
            # set the transition to the initial state of new_start (should always be 0)
            # also, since we are going to append the node "lists," we add the
            # length of the nodes list to the initial state index
            try:
                # try appending
                operand._nodes[accepting_state].transition_dict[""].append(
                    new_start._initial_state
                )
            except KeyError:
                # create a new list on error
                operand._nodes[accepting_state].transition_dict[""] = [
                    new_start._initial_state
                ]

        new_start._nodes.extend(operand._nodes)
        for val in operand._accepting_states:
            val += new_start_inital_num_nodes
            new_start._accepting_states.append(val)

        # combine sigma
        new_sigma = self.__combine_sigma(new_start._sigma, operand._sigma)
        new_start._sigma = new_sigma
        return new_start

    def evaluate_postfix_regex(self, regex):
        """Evaluates the postfix regex and creates an NFA from it.

        Args:
            regex (str): The postfix regular expression.

        Returns:
            NFA: The resulting NFA from evaluating the postfix regular expression.
        """
        stack = []
        for symbol in regex:
            if symbol not in OPERATORS:
                # we have an operand
                stack.append(symbol)
            else:
                # we have an operator
                if symbol == Operator.symbol(Operator.ALTERNATION):
                    # alternation

                    # since we are popping off the stack, we get
                    # rhs then lhs, since it is backwards
                    try:
                        rhs = stack.pop()
                        lhs = stack.pop()
                    except:
                        # error
                        return None
                    stack.append(self.__alternation(lhs, rhs))
                elif symbol == Operator.symbol(Operator.CONCATENATION):
                    # concatenation
                    try:
                        rhs = stack.pop()
                        lhs = stack.pop()
                    except:
                        # error
                        return None
                    stack.append(self.__concatenation(lhs, rhs))
                elif symbol == Operator.symbol(Operator.STAR_CLOSURE):
                    # star closure
                    try:
                        lhs = stack.pop()
                    except:
                        # error
                        return None
                    stack.append(self.__star_closure(lhs))
        # the final answer is the last element in the stack
        return self.__convert_to_nfa(stack.pop())


class Operator(Enum):
    """Enum to represent an operator. Holds the precedence value,
    the number of operands for the operator, and the symbol."""

    # keep track of (precedence value, number of operands, symbol)
    ALTERNATION = (0, 2, "|")
    CONCATENATION = (1, 2, ".")
    STAR_CLOSURE = (2, 1, "*")

    def symbol(operator):
        """Gets the symbol value for this operator.

        Args:
            operator (Operator): The operator.

        Returns:
            str: The operator.
        """
        return operator.value[2]


class PostfixRegex:
    """Contains methods to convert a regular expression from infix to postfix and evaluate it."""

    def __init__(self, infix_regex):
        """Creates a PostfixRegex from a regular expression in infix.

        Args:
            regex (str): The infix regular expression.
        """
        # represent a regular expression as a stack
        self._postfix_regex = self.__infix_to_postfix(infix_regex)

    def get_postfix_regex(self):
        """Gets postfix regular expression instance variable.

        Returns:
            str: The postfix regular expression string.
        """
        return self._postfix_regex

    def generate_nfa(self) -> NFA:
        """Evaluates and parses the regex into an NFA.

        Returns:
            NFA: The output NFA.
        """
        pass

    def __get_precedence(self, operator):
        """Gets the precedence of this operator.

        Args:
            operator (str): The operator symbol.

        Returns:
            int: The operator precedence integer value or None if the operator could not be found.
        """
        if operator == Operator.symbol(Operator.STAR_CLOSURE):
            # since we want the star closure's precedence value
            return Operator.STAR_CLOSURE.value[0]
        elif operator == Operator.symbol(Operator.CONCATENATION):
            return Operator.CONCATENATION.value[0]
        elif operator == Operator.symbol(Operator.ALTERNATION):
            return Operator.ALTERNATION.value[0]
        return None  # bad value

    def __infix_to_postfix(self, regex_str: str) -> str:
        """Converts the regular expression string to postfix from infix.

        Args:
            regex_str (str): The regular expression string.

        Returns:
            str: The output postfix string.
        """
        output = ""
        stack = []
        for char in regex_str:
            if char not in OPERATORS and char not in PARENTHESES:
                # if we have an operand
                output += char
            # check for parentheses
            elif char == "(":
                # push to stack
                stack.append(char)
            elif char == ")":
                # pop from the stack until we find the matching parenthesis
                while stack[len(stack) - 1] != "(":
                    output += stack.pop()
                stack.pop()  # remove the final parenthesis
            else:
                # if we have an operator

                # if the stack is empty or the precedence of the current character
                # is greater than that of the top of the stack or there is a
                # parenthesis in the stack
                if (
                    len(stack) == 0
                    or "(" in stack
                    or self.__get_precedence(char)
                    > self.__get_precedence(stack[len(stack) - 1])
                ):
                    # push this operator onto the stack
                    stack.append(char)
                else:
                    # pop operators from stack until we find one
                    # with a lower precedence (or an empy stack)
                    while len(stack) != 0 and self.__get_precedence(
                        char
                    ) <= self.__get_precedence(stack[len(stack) - 1]):
                        # add popped values to output
                        output += stack.pop()
                    # now we can push this operator to the stack that
                    stack.append(char)

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

        pattern_str = r"(?<=[a-zA-Z*)])(?=[a-zA-Z(])"
        pattern = re.compile(pattern_str)

        match = pattern.search(regex_str)
        while match:
            regex_str = regex_str[: match.start()] + "." + regex_str[match.end() :]
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


#
#
# ---------------------------------------------------------------------------------------------------------
# PART B
#
#
class DFA:
    def __init__(self):
        self.state_list = []
        self.accepting = []
        self.initial = []
        self.sigma = []

    def __str__(self):
        result = " Sigma:\t\t" + "\t".join(self.sigma) + "\n"
        result += " ------------------\n"

        # Add transitions for each state
        for index, state in enumerate(self.state_list):
            transitions = "\t".join(
                str(state.transition_dict.get(symbol, "-")) for symbol in self.sigma
            )
            result += f"     {index}: \t{transitions}\n"

        result += " ------------------\n"

        # Add initial state
        initial_index = (
            self.state_list.index(self.initial)
            if self.initial in self.state_list
            else "-"
        )
        result += f"{initial_index}: Initial State\n"

        # Add accepting states
        accepting_indices = [
            str(self.state_list.index(state)) for state in self.accepting
        ]
        result += ",".join(accepting_indices) + ": Accepting State(s)\n"

        return result

    # gets the closure for a state
    def closure(self, state: Node, input: str, nfa_states: list, visited=None) -> list:
        if visited is None:
            visited = set()
        if state in visited:
            return []
        visited.add(state)
        closure_list: list = state.transition_dict.get(input, [])
        for next_state in closure_list:
            closure_list.extend(
                [
                    value
                    for value in self.closure(
                        nfa_states[next_state], "", nfa_states, visited
                    )
                    if value not in closure_list
                ]
            )
        return closure_list

    # checks if a closure is in a list of closures (this is for nfa to dfa)
    def closure_in_list(self, closure_set: list, closure) -> int:
        for i, existing_closure in enumerate(closure_set):
            if set(existing_closure) == set(closure):
                return i
        return -1

    #
    # TODO: LAMBDA TRANSITITON INCLUDES SELF MAYBE
    #
    # takes a NFA and converts it to a DFA
    def nfa_to_dfa(self, nfa: NFA):
        nfa_states = nfa._nodes
        initial = [0]
        initial.extend(self.closure(nfa_states[0], "", nfa_states))
        state_sets = [initial]
        state_outputs = {}
        index = 0
        alphabet = nfa._sigma

        while index < len(state_sets):
            temp_transitions = {symbol: [] for symbol in alphabet}
            for state in state_sets[index]:
                for symbol in alphabet:
                    temp_transitions[symbol].extend(
                        self.closure(nfa_states[state], symbol, nfa_states)
                    )
            state_outputs[index] = {}
            for symbol in alphabet:
                temp_closure = temp_transitions[symbol]
                state_index = self.closure_in_list(state_sets, temp_closure)
                if state_index == -1:
                    state_sets.append(temp_closure)
                    state_index = len(state_sets) - 1
                state_outputs[index][symbol] = state_index
            index += 1

        # Construct the DFA
        self.sigma = alphabet
        index = 0
        for state_set in state_sets:
            accept = any(nfa_states[state].is_accepting for state in state_set)
            temp_node = Node(state_outputs[index], accept)
            self.state_list.append(temp_node)
            if accept:
                self.accepting.append(temp_node)
            index += 1

        self.initial = self.state_list[0] if self.state_list else None

    #
    #
    # ---------------------------------------------------------------------------------
    # PART C
    #
    #

    # Minimises the DFA
    def minimize_dfa(self):
        # Creating a table for determining distinguishability
        num_states = len(self.state_list)
        distinguishable = [[False] * num_states for _ in range(num_states)]

        # Step 1: Mark where p and q are distinguishible
        accepting_set = set(self.accepting)
        for q in range(num_states):
            for p in range(q):
                if (self.state_list[q] in accepting_set) != (
                    self.state_list[p] in accepting_set
                ):
                    distinguishable[q][p] = True
                    distinguishable[p][
                        q
                    ] = True  # Ensure symmetry on both "sides" of the table

        # Step 2: Iteratively mark pairs as distinguishable
        # we iterate through this loop until no new distinguishable pairs are found
        new_distinguishable_pairs = True
        while new_distinguishable_pairs:
            new_distinguishable_pairs = False
            for q in range(num_states):
                for p in range(q):
                    # Only check undistinguished pairs
                    if not distinguishable[q][p]:
                        # Check if they lead to distinguishable states for any input
                        for symbol in self.sigma:
                            # Handle transitions - in DFA, we should have exactly one target state
                            next_q = self.state_list[q].transition_dict.get(symbol)
                            next_p = self.state_list[p].transition_dict.get(symbol)

                            # If one has a transition and the other doesn't, they're distinguishable
                            if (next_q is None and next_p is not None) or (
                                next_q is not None and next_p is None
                            ):
                                distinguishable[q][p] = True
                                distinguishable[p][q] = True
                                new_distinguishable_pairs = True
                                break

                            # If transitions lead to distinguishable states
                            if next_q != next_p and distinguishable[next_q][next_p]:
                                distinguishable[q][p] = True
                                distinguishable[p][q] = True
                                new_distinguishable_pairs = True
                                break

        # Step 3: Create equivalence classes
        equivalence_classes = []
        assigned = [False] * num_states

        # for every state
        for i in range(num_states):
            # if it is not a part of an equivalence class, create a new class and add the state
            if not assigned[i]:
                new_class = [i]
                assigned[i] = True

                # for every state in the dfa
                for j in range(num_states):
                    # if j is not the same as i, j has not been assigned to a group, and j is distinguishable from i
                    if i != j and not assigned[j] and not distinguishable[i][j]:
                        # we add j to the class, and set j as assigned
                        new_class.append(j)
                        assigned[j] = True
                # add this class to the list of equivalence classes
                equivalence_classes.append(new_class)

        # Step 4: Build the minimized DFA
        minimized_dfa = DFA()
        minimized_dfa.sigma = self.sigma.copy()

        # Create new states for each equivalence class
        for eq_class in equivalence_classes:
            # creating a new state representative of all states in the current EC
            representative_index = eq_class[0]
            is_accepting = self.state_list[representative_index] in self.accepting
            new_state = Node({}, is_accepting)

            # appending the new state to the minimized dfa
            minimized_dfa.state_list.append(new_state)
            if is_accepting:
                minimized_dfa.accepting.append(new_state)

            # Setting initial state
            if self.state_list[representative_index] == self.initial:
                minimized_dfa.initial = new_state

        # Step 5: Mapping state transitions to the minimized dfa
        state_mapping = {}
        for i, eq_class in enumerate(equivalence_classes):
            for state_index in eq_class:
                state_mapping[state_index] = i

        # for each new state
        for i, eq_class in enumerate(equivalence_classes):
            # determine the representative state and it's index
            representative_index = eq_class[0]
            representative = self.state_list[representative_index]

            for symbol in self.sigma:
                if symbol in representative.transition_dict:
                    # get the original destination from the dfa
                    next_state_index = representative.transition_dict[symbol]

                    # map index to corresponding index in the new state
                    new_next_state_index = state_mapping[next_state_index]
                    # add the transition to the minimized dfa
                    minimized_dfa.state_list[i].transition_dict[
                        symbol
                    ] = new_next_state_index

        return minimized_dfa

    # Reads all strings from a given input file and returns the strings that are accepted by the DFA
    def accept_strings(self, input_file):
        result = ""
        string_num = 0

        with open(input_file) as file:
            for line in file:
                strings = line.strip().split()
                for string in strings:
                    steps = list(string)

                    # Start with the index of the initial state
                    curState = self.state_list.index(self.initial)

                    for step in steps:
                        # Check if the step exists as part of the sigma
                        if step not in self.state_list[curState].transition_dict:
                            # Reject the string if the step is not valid
                            curState = None
                            break

                        # Get the index of the next state
                        curState = self.state_list[curState].transition_dict[step]

                    # Check if curState is valid and if the final state is an accepting state, if so add it to the accepted strings
                    if (
                        curState is not None
                        and self.state_list[curState] in self.accepting
                    ):
                        result += "     " + str(string_num) + ":" + string + "\n"
                    string_num += 1
        return result


#
#
# ----------------------------------------------------------------------------------
# MAIN
#
#


def main():
    # take arguments
    if len(sys.argv) != 3:
        # invalid number of arguments
        print("Usage: {} <regex> <file_name>".format(sys.argv[0]))
        sys.exit(1)
    regex_str = sys.argv[1]
    file_name = sys.argv[2]

    # validate regular expression
    (is_valid, regex_str) = RegexValidator.validate_and_modify_regex(regex_str)
    if not is_valid:
        # invalid regular expression
        print("{} is not a valid regular expression".format(regex_str))
        sys.exit(1)

    postfix_regex = PostfixRegex(regex_str)

    nfa = NFA()
    nfa = nfa.evaluate_postfix_regex(postfix_regex.get_postfix_regex())

    if nfa != None:
        print(regex_str, "is a valid regular expression\n")
        print(nfa)
    else:
        print(regex_str, "is not a valid regular expession")
        sys.exit(1)  # terminate program, this is not valid

    # Part B
    dfa = DFA()
    dfa.nfa_to_dfa(nfa)
    print("\nDFA:")
    print(dfa)

    # Part C
    min_dfa = dfa.minimize_dfa()
    print("Minimized DFA:")
    print(min_dfa)

    print("L(" + regex_str + ")")
    print("Accepted strings in " + file_name + ":")
    print(min_dfa.accept_strings(file_name))



if __name__ == "__main__":
    main()
