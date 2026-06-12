input_names = ["timestep", "text_embeds", "time_ids"]
input_specs = dict(timestep=((1, ), "float16"), text_embeds=((1, 1280), "float32"), time_ids=((1, 6), "float16"))
output_names = "silu_3"
compile_options = "--truncate_64bit_io --output_names silu_3"