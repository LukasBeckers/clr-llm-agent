import json

def json_to_dict(json_text):
    """
    Converts a JSON-formatted string to a Python dictionary.

    :param json_text: String containing JSON data
    :return: Python dictionary representation of the JSON data
    :raises: json.JSONDecodeError if the input is not valid JSON
    """
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print("Invalid JSON format:", e)
        return None

