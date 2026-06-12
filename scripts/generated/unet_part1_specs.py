input_names = ["silu_3", "sample", "encoder_hidden_states"]
input_specs = dict(silu_3=((1, 1280), "float16"), sample=((1, 4, 128, 128), "float16"), encoder_hidden_states=((1, 77, 2048), "float16"))
output_names = "add_13,add_2,add_22,add_4,add_55,add_88,conv2d,conv2d_11,conv2d_5"
compile_options = "--truncate_64bit_io --output_names add_13,add_2,add_22,add_4,add_55,add_88,conv2d,conv2d_11,conv2d_5"