import torch
import tqdm


def basic_fed_avg(  # noqa: C901
    nets,
    optimizers,
    criteria,
    dataloaders_train,
    num_rounds,
    batch_samplers,
):
    """Basic implementation of federated averaging. It ensures that all
    networks have the same initialization and the same final value.

    All values are modified in-place, nothing to return.

    Args:
        nets (List[torch.nn.Module]): List of pytorch module.
        optimizers (List[torch optimizers]): List of pytorch optimizers (1 per client).
        criteria (List[torch losses]): List of pytorch losses.
        dataloaders_train (List[torch.utils.data.DataLoader]): List of local data loaders (overrides DataLoaders)
        num_rounds (int): number of rounds
        batch_samplers (List): List of batch samplers used in the train data loaders

    Returns:
        List[float]: time for each round
    """
    num_clients = len(nets)
    # Ensure all networks have the same initialization
    for net in nets[1:]:
        for p, p_ref in zip(net.parameters(), nets[0].parameters()):
            p.data = torch.clone(p_ref.data)

    # Placeholder for sent deltas
    deltas_sent = [None for _ in range(num_clients)]
    aggregated_delta_weights = None
    # Run the loop
    for idx_round in tqdm.tqdm(range(num_rounds), desc="Rounds: "):
        # Simulating "in parallel local training"
        for idx_client in range(num_clients):
            # Update the weights with the aggregated delta
            if aggregated_delta_weights is not None:
                for p, delta in zip(nets[idx_client].parameters(), aggregated_delta_weights):
                    p.data += delta

            # Save the old weights
            old_weights = [torch.clone(p.data) for p in nets[idx_client].parameters()]

            # Run local steps
            for X, y in dataloaders_train[idx_client]:
                optimizers[idx_client].zero_grad()
                y_pred = nets[idx_client](X).reshape(-1)
                loss = criteria[idx_client](y_pred, y)
                loss.backward()
                optimizers[idx_client].step()

            # Get aggregated deltas
            deltas_sent[idx_client] = [
                torch.clone(p_new - p_old).detach()
                for (p_new, p_old) in zip(nets[idx_client].parameters(), old_weights)
            ]

            # Reset local network
            for p_new, p_old in zip(nets[idx_client].parameters(), old_weights):
                p_new.data = p_old

        # Reset the iterator
        for idx_client in range(num_clients):
            batch_samplers[idx_client].reset_counter()

        # Aggregation step
        aggregated_delta_weights = [None for _ in range(len(deltas_sent[0]))]
        for idx_weight in range(len(deltas_sent[0])):
            aggregated_delta_weights[idx_weight] = sum(
                [deltas_sent[idx_client][idx_weight] for idx_client in range(num_clients)]
            )
            aggregated_delta_weights[idx_weight] /= float(num_clients)

        # Update the weights if this is the very last round
        if idx_round == num_rounds - 1:
            for idx_client in range(num_clients):
                for p, delta in zip(nets[idx_client].parameters(), aggregated_delta_weights):
                    p.data += delta
