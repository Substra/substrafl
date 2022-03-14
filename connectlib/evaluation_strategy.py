from typing import List
from typing import Optional
from typing import Union

from connectlib.nodes.test_data_node import TestDataNode


class EvaluationStrategy:
    def __init__(
        self,
        test_data_nodes: List[TestDataNode],
        rounds: Union[int, List[int]],
    ) -> None:
        """Creates an iterator which returns True or False depending on the defined strategy.

        Args:
            test_data_nodes (List[TestDataNode]): nodes on which the model is to be tested.
            rounds (Union[int, List[int]]): on which round the model is to be tested. If rounds is an int the model
                will be tested every ``rounds`` rounds starting from the first round. It will also be tested on the last
                round. If rounds is a list the model will be tested on the index of a round given in the rounds list.
                Note that the first round starts at 1.

        Raises:
            ValueError: test_data_nodes cannot be an empty list
            TypeError: test_data_nodes must be filled with instances of TestDataNode
            TypeError: rounds must be a list or an int
            ValueError: if rounds is an int it must be strictly positive
            ValueError: if rounds is a list it cannot be empty
            TypeError: if rounds is a list it must be filled with variables of type int
            ValueError: if rounds is a list it must be filled only with positive integers

        Example:

            Evaluation strategy which returns True every 2 rounds

            .. code-block:: python

                my_evaluation_strategy = EvaluationStrategy(
                        test_data_nodes = list_of_test_data_nodes,
                        rounds = 2,
                    )

            every next ``next(my_evaluation_strategy)`` will return:

            .. code-block:: python

                True
                False
                True
                True
                StopIteration Error

        Example:

            Evaluation strategy which returns True on rounds 1 and 2

            .. code-block:: python

                my_evaluation_strategy = EvaluationStrategy(
                        test_data_nodes = list_of_test_data_nodes,
                        rounds = [1, 2],
                    )

            every next ``next(my_evaluation_strategy)`` will return

            .. code-block:: python

                True
                True
                False
                False
                False
                StopIteration Error
        """
        self._current_round = 0
        self.test_data_nodes = test_data_nodes
        self._rounds = rounds
        self._num_rounds = None

        if not test_data_nodes:
            raise ValueError("test_data_nodes lists cannot be empty")

        if not all(isinstance(node, TestDataNode) for node in test_data_nodes):
            raise TypeError("test_data_nodes must include objects of TestDataNode type")

        if not isinstance(rounds, (list, int)):
            raise TypeError(f"rounds must be of type list of ints or an int, {type(rounds)} found")

        if isinstance(rounds, int):
            if rounds <= 0:
                raise ValueError(f"rounds must be positive, rounds={rounds} found")
        elif isinstance(rounds, list):
            if not rounds:
                raise ValueError(f"rounds cannot be empty, rounds={rounds} found")
            elif not all(isinstance(r, int) for r in rounds):
                raise TypeError(f"rounds must be of type list of ints or int, {type(rounds)} found")
            elif not all(r > 0 for r in rounds):
                raise ValueError(f"rounds can be only positive, rounds={rounds} found")

    @property
    def num_rounds(self):
        """Property to get the num_rounds.

        Returns:
             Union[int, None]: Total number of rounds.
        """
        return self._num_rounds

    @num_rounds.setter
    def num_rounds(self, num_rounds: int = None):
        """Sets number of rounds (num_rounds) and checks if it is consistent with current evaluation strategy.
        The generator will be reset to the initial state each time num_rounds is set.

        Args:
            num_rounds (Optional[int]): Total number of rounds. If None the iterator may be called
                infinitely. If num_rounds is set the StopIteration Error will be raised if number of calls to next()
                exceeds num_rounds. Defaults to None.
        """
        if num_rounds is not None:
            self._check_rounds_consistency(num_rounds)
        self._num_rounds = num_rounds
        self.restart_rounds()

    def restart_rounds(self):
        """reinitializes current round to 0 (generator will start from the beginning)"""
        self._current_round = 0

    def _check_rounds_consistency(self, num_rounds: Optional[int] = None):
        """Checks if the EvaluationStrategy is consistent with the number of rounds (num_rounds). If num_rounds was
        defined at init this function will be called at init.

        Args:
            num_rounds (Optional[int]): Total number of rounds. If None the iterator may be called
                infinitely. If num_rounds is set the StopIteration Error will be raised if number of calls to next()
                exceeds num_rounds. Defaults to None.

        Raises:
            ValueError: num_rounds must not be smaller than self._rounds if self._rounds is an int
            ValueError: num_rounds must not be be smaller than the largest value of self._rounds
        """

        if isinstance(self._rounds, int) and num_rounds < self._rounds:
            raise ValueError(
                f"Not possible to test every {self._rounds} rounds (from EvaluationStrategy) "
                f"as total number of rounds (num_rounds) from the Strategy is {num_rounds}."
            )
        elif isinstance(self._rounds, list) and max(self._rounds) > num_rounds:
            # rounds is a list of round indices
            raise ValueError(
                f"Not possible to test rounds (from EvaluationStrategy.rounds) greater than {num_rounds}"
                "(num_rounds) from the Strategy."
            )

    def __iter__(self):
        """Required methods for iterables."""
        return self

    def __next__(self):
        """returns True if this round matches the defined strategy and False otherwise.
        required for iterators"""

        test_it = False
        if isinstance(self._rounds, int):
            # checks if _current_round is divisible by rounds or it's a last round
            test_it = (self._current_round % self._rounds == 0) or (self._current_round + 1 == self.num_rounds)
        else:
            # rounds is a list of round indices
            test_it = self._current_round + 1 in self._rounds

        if self.num_rounds and self._current_round + 1 > self.num_rounds:
            # raise an error if number of call to next() exceeded num_rounds
            raise StopIteration(f"Call to the iterator exceeded num_rounds set as {self.num_rounds}")

        self._current_round += 1
        return test_it
