from terminalai.cli import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.shell == "powershell"
    assert args.goal is None
    assert args.model is None
    assert args.max_steps == 20
