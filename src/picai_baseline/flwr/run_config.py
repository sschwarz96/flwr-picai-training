import json


class RunConfig:
    def __init__(self):
        # Data I/O + Experimental Setup
        self.validate_n_epochs = 1
        self.resume_training = 0
        self.overviews_dir = "/home/zimon/flwr-picai-training/workdir/results/UNet/overviews/Task2203_picai_baseline"

        # Training Hyperparameters
        self.image_shape = [16, 128, 128]  # (z, y, x)
        self.num_channels = 3
        self.num_classes = 2
        self.base_lr = 0.0002 # 0.002
        self.focal_loss_gamma = 1.0 # 1.0
        self.enable_da = False  # Data Augmentation
        self.random_seed = 42  # For reproducibility

        # Neural Network-Specific Hyperparameters
        self.model_type = "unet"
        self.model_strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
        self.model_features = [4, 8, 16, 32, 64]
        self.batch_size = 5
        self.use_def_model_hp = 1

        # Federated Learning Config
        self.num_train_epochs = 2
        self.central_evaluation = True
        self.num_clients = 3
        self.num_rounds = 20
        self.num_gpus = 1.0
        self.num_threads_clients = 3
        self.num_threads_augmenting = 2
        self.fraction_fit = 1.0
        self.evaluate_fit = 0.0 if self.central_evaluation else 1.0
        self.folds = [0, 1, 2] if self.central_evaluation else [0, 1, 2, 3, 4]
        self.evaluation_fold = 3 if self.central_evaluation else None

    def to_dict(self):
        """Convert the class attributes to a dictionary for JSON serialization."""
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        """Load configuration from a dictionary."""
        instance = cls()
        for key, value in config_dict.items():
            setattr(instance, key, value)
        return instance

    def to_json(self, filepath):
        """Save the configuration as a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, filepath):
        """Load the configuration from a JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Create an instance


run_configuration = RunConfig()
