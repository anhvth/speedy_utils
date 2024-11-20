import argparse
import yaml
from dataclasses import dataclass, fields, is_dataclass
from typing import Type, TypeVar, Any, Dict
from tabulate import tabulate

T = TypeVar("T")

def dataclass_parser(cls: Type[T]) -> Type[T]:
    """A decorator to add parsing functionality to dataclasses."""
    cls = dataclass(cls)

    def get_parser() -> argparse.ArgumentParser:
        """Generate an argument parser from the dataclass fields."""
        parser = argparse.ArgumentParser(description=f"Parser for {cls.__name__}")
        parser.add_argument("--yaml_file", type=str, help="Path to YAML file with arguments")

        for field in fields(cls):
            arg_name = f"--{field.name}"
            default = field.default
            field_type = field.type
            if field_type is bool:
                parser.add_argument(arg_name, action="store_true", help=f"Enable {field.name} (default: {default})")
            else:
                parser.add_argument(arg_name, type=field_type, default=None, help=f"Override {field.name} (default: {default})")
        
        return parser

    def from_args(args: argparse.Namespace) -> T:
        """Create an instance of the dataclass from argparse arguments."""
        config: Dict[str, Any] = {}
        if args.yaml_file:
            with open(args.yaml_file, "r") as file:
                config = yaml.safe_load(file)

        cli_overrides = {field.name: getattr(args, field.name) for field in fields(cls) if getattr(args, field.name) is not None}
        config.update(cli_overrides)

        return cls(**config)

    def parse_args() -> T:
        """Parse arguments and return an instance of the dataclass."""
        parser = get_parser()
        args = parser.parse_args()
        return from_args(args)

    def cls_str(self):
        return tabulate([[f.name, getattr(self, f.name)] for f in fields(self)], headers=["Field", "Value"], tablefmt="github")

    setattr(cls, "get_parser", staticmethod(get_parser))
    setattr(cls, "from_args", staticmethod(from_args))
    setattr(cls, "parse_args", staticmethod(parse_args))
    cls.__str__ = cls_str

    return cls

@dataclass_parser
class ExampleArgs:
    from_peft: str = "./outputs/llm_hn_qw32b/hn_results_r3/"
    model_name_or_path: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    use_fp16: bool = False
    batch_size: int = 1
    max_length: int = 512
    cache_dir: str = ".cache/run_embeds"
    output_dir: str = ".cache"
    input_file: str = ".cache/doc.csv"
    output_name: str = "qw32b_r3"

if __name__ == "__main__":
    args = ExampleArgs.parse_args()
    print(args)