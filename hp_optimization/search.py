from training import train


def hyperparameter_search(
    dataset_fn,
    model_fn,
    optimizer_fn,
    hyperparameter_configs,
    criterion_fn,
    device,
    epochs,
    batch_size,
    scheduler_fn=None,
):
    train_loader, val_loader, _ = dataset_fn(
        batch_size=batch_size
    )

    results = {}

    for config in hyperparameter_configs:

        print("\n" + "=" * 60)
        print(f"Testing configuration: {config}")
        print("=" * 60)

        model = model_fn().to(device)

        criterion = criterion_fn()

        optimizer = optimizer_fn(
            model=model,
            **config,
        )

        scheduler = None

        if scheduler_fn is not None:
            scheduler = scheduler_fn(
                optimizer=optimizer,
                epochs=epochs,
            )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            scheduler=scheduler,
        )

        results[tuple(config.items())] = {
            "best_val_accuracy": (
                history["model"]["best_val_accuracy"]
            ),
            "training_time": (
                history["optimizer"]["training_time"]
            ),
            "n_forwards": (
                history["optimizer"]["n_forwards"]
            ),
            "n_backwards": (
                history["optimizer"]["n_backwards"]
            ),
            "history": history,
        }

    return results
