import json
from pathlib import Path
from collections import defaultdict

BASE_PATH = Path('/home/zimon/flwr-picai-training/outputs/final_results')


def retrieve_values(folder_name: Path) -> [dict]:
    epsilon_path = BASE_PATH / folder_name
    value_array = []
    for folder in epsilon_path.iterdir():
        if folder.is_dir():
            file_path = epsilon_path / folder / Path('results.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
                value_array.append(data['central_evaluate'])

    return value_array


def get_merged_rounds(value_arrays: [[dict]]) -> [[dict]]:
    merged_round_array = [[] for idx in range(len(value_arrays[0]))]
    for array in value_arrays:
        for idx in range(len(array)):
            merged_round_array[idx].append(array[idx])

    return merged_round_array


def get_averaged_values(merged_round_arrays: [[dict]]):
    averaged_rounds = []
    for array in merged_round_arrays:
        # use defaultdict to auto-zero missing keys
        averaged_values = defaultdict(float)

        for idx in range(len(array)):
            for key, value in array[idx].items():
                if "round" in key:
                    continue

                if "f2" in key:
                    averaged_values = extract_f2_values(averaged_values, value)
                    continue

                # no more try/except needed
                averaged_values[key] += value

        averaged_values = average_values(averaged_values, len(array))
        averaged_rounds.append(averaged_values)

    return averaged_rounds


def average_values(value_dict: dict, nr_values: int) -> dict:
    for key, val in value_dict.items():
        value_dict[key] = val / nr_values

    return value_dict


def extract_f2_values(averaged_values, value_dict):
    # now that averaged_values is a defaultdict, we can drop KeyError handling
    for new_key, item in value_dict.items():
        averaged_values[new_key] += item

    return averaged_values


def save_average_as_json(averaged_rounds, folder_name):
    file_name_path = BASE_PATH / folder_name / Path("averaged_results.json")
    with open(file_name_path, 'w') as f:
        json.dump(averaged_rounds, f)


for path in BASE_PATH.iterdir():
    value_arrays = retrieve_values(path)
    merged_rounds_array = get_merged_rounds(value_arrays)
    averaged_rounds = get_averaged_values(merged_rounds_array)
    for idx, d in enumerate(averaged_rounds):
        averaged_rounds[idx] = {'round': idx + 1, **d}
    save_average_as_json(averaged_rounds, path)