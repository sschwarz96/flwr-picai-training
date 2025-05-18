import json
from pathlib import Path


class RunConfig:
    def __init__(
            self,
            filepath: str = "/home/zimon/flwr-picai-training/src/picai_baseline/flwr/run_config.json",
            **overrides,
    ):
        """
        Load all config fields from the given JSON file, then apply any overrides.

        :param filepath: Path to your JSON config.
        :param overrides: Any config keys you want to override at init.
                          E.g. RunConfig(epsilon=1.0) will set self.epsilon = 1.0
        """


        # 1) Load JSON into this instance
        with open(filepath, "r") as f:
            data = json.load(f)

        for key, val in data.items():
            setattr(self, key, val)

        # 2) Apply overrides
        for key, val in overrides.items():
            setattr(self, key, val)

        print(f"SELLFF {self}")

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
