import inspect
import json
import numpy as np
import os

from types import SimpleNamespace

class SchedulerMixin:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__dict__["config"] = SimpleNamespace(**kwargs)
        return instance

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        subfolder=None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        config = cls.load_config(pretrained_model_name_or_path)
        return cls.from_config(config, **kwargs)


class ConfigMixin:
    @classmethod
    def load_config(cls, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
            
    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        signature = inspect.signature(cls.__init__)
        expected_keys = set(dict(signature.parameters).keys())
        expected_keys.remove("self")

        init_dict = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0
        }
        
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")

        for key in expected_keys:
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)

            if key in kwargs:
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                init_dict[key] = config_dict.pop(key)

        return init_dict


    @classmethod
    def from_config(cls, config=None, return_unused_kwargs=False, **kwargs):
        init_dict = cls.extract_init_dict(config, **kwargs)
        model = cls(**init_dict)

        return model

def register_to_config(init):
    return init
