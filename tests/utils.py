import pickle

from substrafl.nodes.node import OutputIdentifiers
from substrafl.schemas import TaskType


def download_train_task_models_by_rank(network, session_dir, my_algo, compute_plan, rank: int):
    # Retrieve local train task key
    train_tasks = network.clients[0].list_task(
        filters={
            "compute_plan_key": [compute_plan.key],
            "rank": [rank],
        }
    )
    local_models = list()
    for task in train_tasks:
        client = None
        if task.worker == network.msp_ids[0]:
            client = network.clients[0]
        elif task.worker == network.msp_ids[1]:
            client = network.clients[1]

        outputs = client.list_task_output_assets(task.key)
        for output in outputs:
            if output.identifier != OutputIdentifiers.local:
                continue
            model_path = client.download_model(output.asset.key, session_dir)
            model = my_algo.load_local_state(model_path)
            # Move the torch model to CPU
            model.model.to("cpu")
            local_models.append(model)
    return local_models


def download_aggregate_model_by_rank(network, session_dir, compute_plan, rank: int):
    aggregate_tasks = network.clients[0].list_task(filters={"compute_plan_key": [compute_plan.key], "rank": [rank]})
    aggregate_tasks = [t for t in aggregate_tasks if t.tag == TaskType.AGGREGATE]
    assert len(aggregate_tasks) == 1
    model_path = network.clients[0].download_model_from_task(
        aggregate_tasks[0].key, identifier=OutputIdentifiers.shared, folder=session_dir
    )
    aggregate_model = pickle.loads(model_path.read_bytes())

    return aggregate_model
