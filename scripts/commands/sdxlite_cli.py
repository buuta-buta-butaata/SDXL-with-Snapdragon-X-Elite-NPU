import logging
import os
from argparse import ArgumentParser

from .txt2img import Txt2ImgCommand
from .setup import SetupCommand
from utils import get_project_root, console_colored_filter

def main():
    root_dir = get_project_root()
    os.chdir(root_dir)
    
    parser = ArgumentParser(prog="sdxlite_cli")
    commands_parser = parser.add_subparsers(title="Commands", metavar="<command>")

    Txt2ImgCommand.register_subcommand(commands_parser)
    SetupCommand.register_subcommand(commands_parser)

    args = parser.parse_args()
    logging.basicConfig(
        # format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        format='[%(levelname)s] %(name)s: %(message)s',
        level=args.log_level if hasattr(args, "log_level") else logging.INFO
    )

    for handler in logging.root.handlers:
        handler.addFilter(console_colored_filter)
        
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
