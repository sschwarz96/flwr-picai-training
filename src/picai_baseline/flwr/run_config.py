import json


class RunConfig:
    def __init__(self):
        # Data I/O + Experimental Setup
        self.max_threads = 6
        self.validate_n_epochs = 1
        self.validate_min_epoch = 0
        self.export_best_model = 1
        self.resume_training = 1
        self.weights_dir = "/home/zimon/picai_baseline/workdir/results/UNet/weights"
        self.overviews_dir = "/home/zimon/picai_baseline/workdir/results/UNet/overviews/Task2203_picai_baseline"
        self.folds = [0, 1, 2, 3, 4]

        # Training Hyperparameters
        self.image_shape = [20, 256, 256]  # (z, y, x)
        self.num_channels = 3
        self.num_classes = 2
        self.num_train_epochs = 1
        self.base_lr = 0.001
        self.focal_loss_gamma = 1.0
        self.enable_da = 1  # Data Augmentation

        # Neural Network-Specific Hyperparameters
        self.model_type = "unet"
        self.model_strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
        self.model_features = [32, 64, 128, 256, 512, 1024]
        self.batch_size = 4
        self.use_def_model_hp = 1

        # Federated Learning Config
        self.num_clients = 3
        self.num_rounds = 5

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
