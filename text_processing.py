def processing(input_string: str):
    output_string = ' '.join(input_string.split()).replace('»', '', 1).strip().capitalize()
    while not (output_string.endswith('.') or output_string.endswith('!')):
        output_string = output_string[:-1]
    return output_string