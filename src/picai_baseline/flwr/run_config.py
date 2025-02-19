class RunConfig:
    # Data I/O + Experimental Setup
    max_threads = 6
    validate_n_epochs = 1
    validate_min_epoch = 0
    export_best_model = 1
    resume_training = 0  # Assuming it should be an int, change to str if needed
    weights_dir = "/home/zimon/picai_baseline/workdir/results/UNet/weights"  # Required, default to an empty string
    overviews_dir = "/home/zimon/picai_baseline/workdir/results/UNet/overviews/Task2203_picai_baseline"  # Required, default to an empty string
    folds = [0, 1, 2, 3, 4]  # Assuming a list of integers

    # Training Hyperparameters
    image_shape = [20, 256, 256]  # (z, y, x)
    num_channels = 3
    num_classes = 2
    num_train_epochs = 1
    base_lr = 0.001
    focal_loss_gamma = 1.0
    enable_da = 1  # Data Augmentation

    # Neural Network-Specific Hyperparameters
    model_type = "unet"
    model_strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]  # Converted from string
    model_features = [32, 64, 128, 256, 512, 1024]  # Converted from string
    batch_size = 4
    use_def_model_hp = 1

    #Federated learning config
    num_clients = 3
    num_rounds=5



run_configuration = RunConfig()
