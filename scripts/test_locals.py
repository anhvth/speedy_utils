from speedy_utils import multi_process


def process_data(item):
    # Mix of user variables and imported modules
    data = {"value": item}
    multiplier = 2

    # This will cause an error
    denominator = 0
    final = data["value"] * multiplier / denominator
    return final


if __name__ == "__main__":
    inputs = range(5)
    multi_process(process_data, inputs, backend="mp")
