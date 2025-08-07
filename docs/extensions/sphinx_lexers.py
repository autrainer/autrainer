from typing import Any, Dict

from pygments.lexer import RegexLexer, bygroups
from pygments.token import (
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    String,
    Text,
)
from sphinx.application import Sphinx
from sphinx.highlighting import lexers


SUBCOMMANDS = [
    "create",
    "list",
    "show",
    "fetch",
    "preprocess",
    "train",
    "postprocess",
    "rm-failed",
    "rm-states",
    "inference",
    "group",
]


class AutrainerLexer(RegexLexer):
    name = "autrainer"
    aliases = ["autrainer", "aucli"]
    filenames = []

    tokens = {
        "root": [
            (
                rf"(autrainer)(\s+)({'|'.join(SUBCOMMANDS)})",
                bygroups(Name.Builtin, Text, Keyword),
            ),
            (r"\b(autrainer)\b", Name.Builtin),
            (r"--\w+", Operator),
            (r"=\S+", Literal),
            (r"-\w+", Operator),
            (r"[^\s]+", Text),
            (r"\s+", Text),
        ]
    }


class PipLexer(RegexLexer):
    name = "pip"
    aliases = ["pip"]
    filenames = []

    tokens = {
        "root": [
            (
                r"(pip)(\s+)(install|uninstall|freeze|list|show|search)",
                bygroups(Name.Builtin, Text, Keyword),
            ),
            (r"(--\w+)", Operator),
            (r"(-\w+)", Operator),
            (
                r"([a-zA-Z0-9_-]+)(\[.*?\])?",
                bygroups(Name.Function, Name.Variable),
            ),
            (r"==|>=|<=|!=|~=|>", Operator),
            (r"\d+\.\d+\.\d+", Number),
            (r"[^-\s]+", Text),
            (r"\s+", Text),
        ]
    }


EXTENDED_COMMANDS = [
    "git",
    "cd",
    "uv",
    "pre-commit",
    "pytest",
    "mkdir",
    "mlflow",
    "tensorboard",
]


class ExtendedLexer(RegexLexer):
    name = "extended"
    aliases = ["extended"]
    filenames = []

    tokens = {
        "root": [
            (rf"\b({'|'.join(EXTENDED_COMMANDS)})\b", Keyword),
            (r"(&&|\|\||;)", Operator),
            (r"(--\w+)", Operator),
            (r"(-\w+)", Operator),
            (r"(\b\w+@[\w.]+\b)", Name.Variable),
            (r"(/[^\s]*)", Name.Variable),
            (r'(".*?"|\'.*?\')', String),
            (r"\b\d+\b", Number),
            (r"[^-\s]+", Text),
            (r"\s+", Text),
        ]
    }


def setup(app: Sphinx) -> Dict[str, Any]:
    lexers["autrainer"] = AutrainerLexer(startinline=True)
    lexers["aucli"] = AutrainerLexer(startinline=True)
    lexers["pip"] = PipLexer(startinline=True)
    lexers["extended"] = ExtendedLexer(startinline=True)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
