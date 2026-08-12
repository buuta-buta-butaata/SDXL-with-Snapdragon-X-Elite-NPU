from abc import ABC, abstractmethod
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter

class BaseCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

import argparse
from argparse import (OPTIONAL, SUPPRESS, ZERO_OR_MORE,
                      ArgumentDefaultsHelpFormatter, ArgumentParser,
                      RawDescriptionHelpFormatter, RawTextHelpFormatter)


# https://qiita.com/yuji38kwmt/items/c7c4d487e3188afd781e
class MyHelpFormatter(
    RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
):
    def _get_help_string(self, action):
        help = action.help
        if action.required:
            help += " (required)"

        if "%(default)" not in action.help:
            if action.default is not SUPPRESS:
                defaulting_nargs = [OPTIONAL, ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    if action.default is not None and not action.const:
                        if action.default == "":
                            help += "\n(default: '')"
                        else:
                            help += "\n(default: %(default)s)"
        return help

def int_range(lower_limit, upper_limit):
    def checker(value):
        ivalue = int(value)
        if ivalue < lower_limit or ivalue > upper_limit:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {value}. Must be between {lower_limit} and {upper_limit}."
            )
        return ivalue
    return checker
    
def format_dict_text(d: dict):
    return "\n".join([
        f"  {k}: '{v}'" for k, v in d.items()
    ])
