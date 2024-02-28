from typing import List
from typing import Optional
from typing import Set

from substrafl.nodes import TestDataNodeProtocol


class EvaluationStrategy:
    def __init__(
        self,
        test_data_nodes: List[TestDataNodeProtocol],
        eval_frequency: Optional[int] = None,
        eval_rounds: Optional[List[int]] = None,
    ) -> None:
        """Creates an iterator which returns True or False depending on the defined strategy.
        At least one of eval_frequency or eval_rounds must be defined. If both are defined, the
        union of both selected indexes will be evaluated.

        Args:
            test_data_nodes (List[TestDataNodeProtocol]): nodes on which the model is to be tested.
            eval_frequency (Optional[int]): The model will be tested every ``eval_frequency`` rounds.
                Set to None to activate eval_rounds only. Defaults to None.
            eval_rounds (Optional[List[int]]): If specified, the model will be tested on the index of a round given
                in the rounds list. Set to None to activate eval_frequency only. Defaults to None.

        Raises:
            ValueError: test_data_nodes cannot be an empty list
            TypeError: test_data_nodes must be filled with instances of TestDataNodeProtocol
            TypeError: rounds must be a list or an int
            ValueError: both eval_rounds and eval_frequency cannot be None at the same time

        Example:

            Evaluation strategy which returns True every 2 rounds

            .. code-block:: python

                my_evaluation_strategy = EvaluationStrategy(
                        test_data_nodes = list_of_test_data_nodes,
                        eval_frequency = 2,
                        eval_rounds=None,
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
                        eval_frequency = None,
                        eval_rounds = [1, 2],
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
        self._eval_frequency = eval_frequency
        self._eval_rounds = eval_rounds
        self._num_rounds = None

        if not test_data_nodes:
            raise ValueError("test_data_nodes lists cannot be empty")

        if not all(isinstance(node, TestDataNodeProtocol) for node in test_data_nodes):
            raise TypeError("test_data_nodes must implement the TestDataNodeProtocol")

        if eval_frequency is None and eval_rounds is None:
            raise ValueError("At least one of eval_frequency or eval_rounds must be defined")

        self._check_eval_frequency()
        self._check_eval_rounds()

    @property
    def test_data_nodes_org_ids(self) -> Set:
        """Property to get the ids or test data nodes organizations.

        Returns:
             Set: set of organization ids
        """
        return {test_data_node.organization_id for test_data_node in self.test_data_nodes}

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
            ValueError: num_rounds must not be smaller than self._eval_frequency
            ValueError: num_rounds must not be be smaller than the largest value of self._eval_rounds
        """

        if self._eval_frequency is not None and num_rounds < self._eval_frequency:
            raise ValueError(
                f"Not possible to test every {self._eval_frequency} rounds (from EvaluationStrategy) "
                f"as total number of rounds (num_rounds) from the Strategy is {num_rounds}."
            )

        if self._eval_rounds is not None and max(self._eval_rounds) > num_rounds:
            # rounds is a list of round indices
            raise ValueError(
                f"Not possible to test rounds (from EvaluationStrategy.rounds) greater than {num_rounds}"
                "(num_rounds) from the Strategy."
            )

    def _check_eval_frequency(self):
        """Check eval_frequency validity.

        Raises:
            ValueError: eval_frequency must be a positive int
            TypeError: eval_frequency must be of type int or None
        """
        if isinstance(self._eval_frequency, int):
            if self._eval_frequency <= 0:
                raise ValueError(f"eval_frequency must be positive, eval_frequency={self._eval_frequency} found")
        elif self._eval_frequency is not None:
            raise TypeError(f"eval_frequency must be of type int, {type(self._eval_frequency)} found")

    def _check_eval_rounds(self):
        """Check eval_rounds validity.

        Raises:
            ValueError: if eval_rounds is a list it cannot be empty
            TypeError: if eval_rounds is a list it must be filled with variables of type int
            ValueError: if eval_rounds is a list it must be filled with positive integers
            TypeError: eval_rounds must be of type list or None
        """
        if isinstance(self._eval_rounds, list):
            if not self._eval_rounds:
                raise ValueError(f"eval_rounds cannot be empty, eval_rounds={self._eval_rounds} found")
            elif not all(isinstance(r, int) for r in self._eval_rounds):
                raise TypeError(
                    "eval_rounds must only contains integers,"
                    f" {[type(x) for x in self._eval_rounds if not isinstance(x, int)]} found"
                )
            elif not all(r >= 0 for r in self._eval_rounds):
                raise ValueError(f"eval_rounds can be only positive indexes, rounds={self._eval_rounds} found")
        elif self._eval_rounds is not None:
            raise TypeError(f"eval_rounds must be of type list of ints, {type(self._eval_rounds)} found")

    def __iter__(self):
        """Required methods for iterables."""
        return self

    def __next__(self):
        """returns True if this round matches the defined strategy and False otherwise.
        required for iterators"""

        test_it = False

        if self._eval_frequency is not None:
            # checks if _current_round is divisible by rounds or it's a last round
            test_it = (self._current_round % self._eval_frequency == 0) or (self._current_round == self.num_rounds)

        if self._eval_rounds is not None:
            # rounds is a list of round indices
            test_it = test_it or self._current_round in self._eval_rounds

        if self.num_rounds and self._current_round > self.num_rounds:
            # raise an error if number of call to next() exceeded num_rounds
            raise StopIteration(f"Call to the iterator exceeded num_rounds set as {self.num_rounds}")

        self._current_round += 1
        return test_it
