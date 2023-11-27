from substrafl.nodes.schemas import SimuPerformancesMemory
from substrafl.nodes.schemas import SimuStatesMemory


def test_simu_performance_memory_concatenation():
    perf1 = SimuPerformancesMemory(
        worker=["dummy_1", "dummy2"], round_idx=[1, 1], identifier=["metric1", "metric2"], performance=[0.1, 1.1]
    )
    perf2 = SimuPerformancesMemory(worker=["dummy_1"], round_idx=[15], identifier=["metric3"], performance=[0.5])

    concat = SimuPerformancesMemory(
        worker=["dummy_1", "dummy2", "dummy_1"],
        round_idx=[1, 1, 15],
        identifier=["metric1", "metric2", "metric3"],
        performance=[0.1, 1.1, 0.5],
    )

    assert perf1 + perf2 == concat


def test_simu_states_memory_concatenation():
    state1 = SimuStatesMemory(worker=["dummy_1", "dummy2"], round_idx=[1, 1], state=["state1", "state2"])
    state2 = SimuStatesMemory(worker=["dummy_1"], round_idx=[15], state=["state3"])

    concat = SimuStatesMemory(
        worker=["dummy_1", "dummy2", "dummy_1"], round_idx=[1, 1, 15], state=["state1", "state2", "state3"]
    )

    assert state1 + state2 == concat
