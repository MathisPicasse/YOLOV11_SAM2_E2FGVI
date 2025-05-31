import yaml
from modules.exceptions.custom_exceptions import TrackerTypeError

class Tracker():
    def __init__(self, config_path: str, name_tracker: str):
        """
        Initialize the tracker with parameters loaded from a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
            name_tracker (str): Name of the tracker to initialize ('botsort', 'bytetrack').

        Raises:
            FileNotFoundError: If the YAML configuration file does not exist.
            ValueError: If there is an error while parsing the YAML file.
            TrackerTypeError: If the specified tracker type is not found in the configuration.
        """
        try:
            with open(config_path, 'r') as f:
                # Load the YAML configuration into a dictionary
                tracker_configuration = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_path}' doesn't exist")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML Parsing error in file '{config_path}': {e}")

        # Check if the requested tracker type exists in the loaded configuration
        if name_tracker not in tracker_configuration:
            raise TrackerTypeError(name_tracker)

        # Dynamically set attributes on the Tracker instance based on configuration parameters
        for param, value in tracker_configuration[name_tracker].items():
            setattr(self, param, value)

    def __repr__(self):
        return f"Tracker(type={self.tracker_type}, params={self.__dict__})"


