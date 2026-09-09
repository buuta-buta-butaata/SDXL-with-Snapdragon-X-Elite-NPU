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

def float_range(lower_limit, upper_limit):
    def checker(value):
        fvalue = float(value)
        if fvalue < lower_limit or fvalue > upper_limit:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {value}. Must be between {lower_limit} and {upper_limit}."
            )
        return fvalue
    return checker

def format_dict_text(d: dict):
    return "\n".join([
        f"  {k}: '{v}'" for k, v in d.items()
    ])

# TODO:
def modify_parser_action(parser, option_string, **kwargs):
    """
    指定したパーサーの引数（Action）の属性を動的に上書きする汎用関数
    
    :param parser: 対象の ArgumentParser または Subparser オブジェクト
    :param option_string: 変更したい引数の文字列 (例: "--prompt", "-i")
    :param kwargs: 上書きしたい属性と値のペア (例: help="新しい説明", default=10)
    """
    # parser._actions から、指定された option_string を含む Action を探す
    target_action = None
    for action in parser._actions:
        if option_string in action.option_strings:
            target_action = action
            break
            
    if target_action is None:
        raise ValueError(f"Argument '{option_string}' not found in the parser.")
        
    # 指定された属性を動的に上書き
    for key, value in kwargs.items():
        if hasattr(target_action, key):
            setattr(target_action, key, value)
        else:
            raise AttributeError(f"Action object has no attribute '{key}'")

def remove_parser_action(parser, option_string):
    target_action = None
    for action in parser._actions:
        if option_string in action.option_strings:
            target_action = action
            break
    target_action.container._remove_action(target_action)
