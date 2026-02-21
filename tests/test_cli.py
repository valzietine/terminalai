from terminalai.cli import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.shell == "powershell"
    assert args.model is None
    assert args.dry_run is False
