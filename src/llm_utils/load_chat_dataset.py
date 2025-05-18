from speedy_utils import load_by_ext

from llm_utils.chat_format import (display_chat_messages_as_html,
                                   identify_format, transform_messages)


def load_chat_dataset(path, current_format='auto', return_format='chatml'):
    """
    Load a chat dataset from a given path and convert it to a specified format.

    This function loads the dataset from a given file path, identifies its current
    format if not explicitly provided, and converts it to the desired format if needed.

    Parameters:
    path (str): The path to the dataset file.
    current_format (str, optional): The current format of the dataset. Default is 'auto'.
                                    If set to 'auto', the format will be automatically identified.
                                    Otherwise, it should be one of ['sharegpt', 'chatml'].
    return_format (str, optional): The format to which the dataset should be converted.
                                   Default is 'chatml'. Should be one of ['sharegpt', 'chatml'].

    Returns:
    list: A list of chat messages in the desired format.

    Raises:
    AssertionError: If the return format is not recognized or the data is not in the correct list format.
    """
    
    assert return_format in ['sharegpt', 'chatml'], "The return format is not recognized. Please specify the return format."
    data= load_by_ext(path)
    assert isinstance(data, list), "The data is not in the correct format. Please check the format of the data."
    if current_format == 'auto':
        current_format = identify_format(data[0])
    if current_format != return_format:
        from loguru import logger
        logger.info(f"Converting the {path} from {current_format} to {return_format}.")
        items = [transform_messages(item, current_format, return_format) for item in data]
    else:
        items = data

    return items