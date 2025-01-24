import argparse
from argparse import REMAINDER, ArgumentParser, Namespace

import pydantic

from .argparse_util import check_env, env


class PydanticArgparseWrapper(pydantic.BaseModel):
    parser: ArgumentParser = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.parser = self.__class__.parser

    def to_argparse(self) -> ArgumentParser:
        args_list = []
        positional_args = []
        for action in self.parser._actions:
            if action.dest == "parser":
                continue
            value = getattr(self, action.dest, None)
            if value is None:
                continue
            if action.option_strings:
                if isinstance(value, bool):
                    if value:
                        args_list.append(action.option_strings[0])
                else:
                    args_list.append(action.option_strings[0])
                    args_list.append(str(value))
            elif action.nargs == REMAINDER:
                positional_args.extend(value)
            else:
                positional_args.append(str(value))
        return self.parser.parse_args(args_list + positional_args)


def convert_to_pydantic(argparse_parser: argparse.ArgumentParser) -> type:
    fields = {}
    for action in argparse_parser._actions:
        if action.dest == "help":
            continue
        if action.dest != argparse.SUPPRESS:
            field_type = (
                list[str] if action.nargs else (action.type if action.type else str)
            )
            default_value = (
                action.default if action.default is not argparse.SUPPRESS else ...
            )
            if action.required:
                fields[action.dest] = (field_type, ...)
            else:
                fields[action.dest] = (field_type, default_value)

    model_klass = pydantic.create_model(
        argparse_parser.description, **fields, __base__=PydanticArgparseWrapper
    )
    model_klass.parser = argparse_parser
    return model_klass


# Unit tests
# ----------


def test_empty():
    parser = ArgumentParser(description="empty")
    PydanticArgs = convert_to_pydantic(parser)
    args = PydanticArgs()
    assert isinstance(args.to_argparse(), Namespace)


def test_basic():
    parser = ArgumentParser(description="basic")
    parser.add_argument("--foo", type=str, default="bar")
    parser.add_argument("--baz", type=int, default=1)
    PydanticArgs = convert_to_pydantic(parser)
    # All defaults
    args = PydanticArgs()
    assert args.foo == "bar"
    assert args.baz == 1

    # All specified
    args = PydanticArgs(foo="foo", baz=2)
    assert args.foo == "foo"
    assert args.baz == 2

    # Some specified (partial defaulting)
    args = PydanticArgs(foo="123")
    assert args.foo == "123"
    assert args.baz == 1
    assert isinstance(args, pydantic.BaseModel)
    argparse_args = args.to_argparse()
    assert argparse_args.foo == "123"
    assert argparse_args.baz == 1


def test_positional():
    parser = ArgumentParser(description="positional")
    parser.add_argument("foo", type=str, default="bar")
    parser.add_argument("baz", type=int, default=1)
    PydanticArgs = convert_to_pydantic(parser)
    args = PydanticArgs(foo="123", baz=2)
    assert args.foo == "123"
    assert args.baz == 2
    assert isinstance(args, pydantic.BaseModel)
    argparse_args = args.to_argparse()
    assert argparse_args.foo == "123"
    assert argparse_args.baz == 2


def test_positional_no_defaults():
    parser = ArgumentParser(description="positional_no_defaults")
    parser.add_argument("foo", type=str)
    parser.add_argument("baz", type=int)
    PydanticArgs = convert_to_pydantic(parser)
    args = PydanticArgs(foo="123", baz=2)
    assert args.foo == "123"
    assert args.baz == 2
    assert isinstance(args, pydantic.BaseModel)
    argparse_args = args.to_argparse()
    assert argparse_args.foo == "123"
    assert argparse_args.baz == 2


def test_positionals_missing_error_handling():
    import pytest  # inlined to avoid needing to install outside of testing

    parser = ArgumentParser(description="positional_no_defaults_error_handling")
    parser.add_argument("foo", type=str)
    parser.add_argument("baz", type=int)
    PydanticArgs = convert_to_pydantic(parser)
    with pytest.raises(pydantic.ValidationError):
        PydanticArgs()

    with pytest.raises(pydantic.ValidationError):
        PydanticArgs(baz=123)

    with pytest.raises(pydantic.ValidationError):
        PydanticArgs(foo="123")


def test_positionals_wrong_type_handling():
    parser = ArgumentParser(description="positionals_wrong_type_error_handling")
    parser.add_argument("foo", type=int)
    parser.add_argument("baz", type=str)
    PydanticArgs = convert_to_pydantic(parser)
    args = PydanticArgs(foo="123", baz=123)
    # pydantic 2 has default type coercion behavior: https://docs.pydantic.dev/latest/concepts/types/#type-conversion.
    # Because argparse is string-based, we don't enable 'strict' mode in pydantic.
    assert args.foo == 123
    assert args.baz == "123"
    argparse_args = args.to_argparse()
    assert argparse_args.foo == 123
    assert argparse_args.baz == "123"


def test_torchrun():
    # Taken from https://github.dev/pytorch/pytorch/blob/main/torch/distributed/run.py
    def get_torchrun_args_parser() -> ArgumentParser:
        """Parse the command line options."""
        parser = ArgumentParser(
            description="Torch Distributed Elastic Training Launcher"
        )

        #
        # Worker/node size related arguments.
        #

        parser.add_argument(
            "--nnodes",
            action=env,
            type=str,
            default="1:1",
            help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
        )
        parser.add_argument(
            "--nproc-per-node",
            "--nproc_per_node",
            action=env,
            type=str,
            default="1",
            help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
        )

        #
        # Rendezvous related arguments
        #

        parser.add_argument(
            "--rdzv-backend",
            "--rdzv_backend",
            action=env,
            type=str,
            default="static",
            help="Rendezvous backend.",
        )
        parser.add_argument(
            "--rdzv-endpoint",
            "--rdzv_endpoint",
            action=env,
            type=str,
            default="",
            help="Rendezvous backend endpoint; usually in form <host>:<port>.",
        )
        parser.add_argument(
            "--rdzv-id",
            "--rdzv_id",
            action=env,
            type=str,
            default="none",
            help="User-defined group id.",
        )
        parser.add_argument(
            "--rdzv-conf",
            "--rdzv_conf",
            action=env,
            type=str,
            default="",
            help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
        )
        parser.add_argument(
            "--standalone",
            action=check_env,
            help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
            "on a free port. Useful when launching single-node, multi-worker job. If specified "
            "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
            "are ignored.",
        )

        #
        # User-code launch related arguments.
        #

        parser.add_argument(
            "--max-restarts",
            "--max_restarts",
            action=env,
            type=int,
            default=0,
            help="Maximum number of worker group restarts before failing.",
        )
        parser.add_argument(
            "--monitor-interval",
            "--monitor_interval",
            action=env,
            type=float,
            default=0.1,
            help="Interval, in seconds, to monitor the state of workers.",
        )
        parser.add_argument(
            "--start-method",
            "--start_method",
            action=env,
            type=str,
            default="spawn",
            choices=["spawn", "fork", "forkserver"],
            help="Multiprocessing start method to use when creating workers.",
        )
        parser.add_argument(
            "--role",
            action=env,
            type=str,
            default="default",
            help="User-defined role for the workers.",
        )
        parser.add_argument(
            "-m",
            "--module",
            action=check_env,
            help="Change each process to interpret the launch script as a Python module, executing "
            "with the same behavior as 'python -m'.",
        )
        parser.add_argument(
            "--no-python",
            "--no_python",
            action=check_env,
            help="Skip prepending the training script with 'python' - just execute it directly. Useful "
            "when the script is not a Python script.",
        )

        parser.add_argument(
            "--run-path",
            "--run_path",
            action=check_env,
            help="Run the training script with runpy.run_path in the same interpreter."
            " Script must be provided as an abs path (e.g. /abs/path/script.py)."
            " Takes precedence over --no-python.",
        )
        parser.add_argument(
            "--log-dir",
            "--log_dir",
            action=env,
            type=str,
            default=None,
            help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
            "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
            "rdzv_id as the prefix).",
        )
        parser.add_argument(
            "-r",
            "--redirects",
            action=env,
            type=str,
            default="0",
            help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
            "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
            "stderr for local rank 1).",
        )
        parser.add_argument(
            "-t",
            "--tee",
            action=env,
            type=str,
            default="0",
            help="Tee std streams into a log file and also to console (see --redirects for format).",
        )

        parser.add_argument(
            "--local-ranks-filter",
            "--local_ranks_filter",
            action=env,
            type=str,
            default="",
            help="Only show logs from specified ranks in console (e.g. [--local_ranks_filter=0,1,2] will "
            "only show logs from rank 0, 1 and 2). This will only apply to stdout and stderr, not to"
            "log files saved via --redirect or --tee",
        )

        #
        # Backwards compatible parameters with caffe2.distributed.launch.
        #

        parser.add_argument(
            "--node-rank",
            "--node_rank",
            type=int,
            action=env,
            default=0,
            help="Rank of the node for multi-node distributed training.",
        )
        parser.add_argument(
            "--master-addr",
            "--master_addr",
            default="127.0.0.1",
            type=str,
            action=env,
            help="Address of the master node (rank 0) that only used for static rendezvous. It should "
            "be either the IP address or the hostname of rank 0. For single node multi-proc training "
            "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
            "`[0:0:0:0:0:0:0:1]`.",
        )
        parser.add_argument(
            "--master-port",
            "--master_port",
            default=29500,
            type=int,
            action=env,
            help="Port on the master node (rank 0) to be used for communication during distributed "
            "training. It is only used for static rendezvous.",
        )
        parser.add_argument(
            "--local-addr",
            "--local_addr",
            default=None,
            type=str,
            action=env,
            help="Address of the local node. If specified, will use the given address for connection. "
            "Else, will look up the local node address instead. Else, it will be default to local "
            "machine's FQDN.",
        )

        parser.add_argument(
            "--logs-specs",
            "--logs_specs",
            default=None,
            type=str,
            help="torchrun.logs_specs group entrypoint name, value must be type of LogsSpecs. "
            "Can be used to override custom logging behavior.",
        )

        #
        # Positional arguments.
        #

        parser.add_argument(
            "training_script",
            type=str,
            help="Full path to the (single GPU) training program/script to be launched in parallel, "
            "followed by all the arguments for the training script.",
        )

        # Rest from the training program.
        parser.add_argument("training_script_args", nargs=REMAINDER)

        return parser

    parser = get_torchrun_args_parser()
    args = parser.parse_args(
        ["--nnodes", "1:4", "--nproc-per-node", "1", "main.py", "bar"]
    )
    assert args.nnodes == "1:4"
    assert args.nproc_per_node == "1"
    assert args.training_script == "main.py"
    assert args.training_script_args == ["bar"]

    model = convert_to_pydantic(parser)
    pydantic_args = model(
        nnodes="1:4",
        nproc_per_node="1",
        training_script="main.py",
        training_script_args=["bar"],
    )
    assert pydantic_args.nnodes == "1:4"
    assert pydantic_args.nproc_per_node == "1"
    assert pydantic_args.training_script == "main.py"
    assert pydantic_args.training_script_args == ["bar"]

    argparse_args = pydantic_args.to_argparse()
    assert argparse_args.nnodes == "1:4"
    assert argparse_args.nproc_per_node == "1"
    assert argparse_args.training_script == "main.py"
    assert argparse_args.training_script_args == ["bar"]
