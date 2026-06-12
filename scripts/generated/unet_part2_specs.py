input_names = ["add_88", "silu_3", "encoder_hidden_states"]
input_specs = dict(add_88=((1, 1280, 32, 32), "float16"), silu_3=((1, 1280), "float16"), encoder_hidden_states=((1, 77, 2048), "float16"))
output_names = "add_123"
compile_options = "--truncate_64bit_io --output_names add_123"