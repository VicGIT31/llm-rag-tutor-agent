"""Duolingoy: python -m sensai.web [--port 8000] [options of python -m sensai: --model, --user, --config…]"""

import argparse
import sys

import uvicorn

from ..config import ConfigError, load_config
from ..ollama_client import OllamaError
from .app import create_app
from .engine import Engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sensai.web",
        description="Duolingoy : Sensai dans le navigateur. Accepte aussi les options de « python -m sensai » "
        "(--model, --user, --config, --num-ctx, --db, --host).",
    )
    parser.add_argument("--port", type=int, default=8000, help="port local de l'interface (défaut : 8000)")
    args, rest = parser.parse_known_args(argv)
    try:
        config = load_config(rest)
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    try:
        engine = Engine(config)
    except OllamaError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print(f"Duolingoy — modèle {config.model}, apprenant « {config.user} » : http://127.0.0.1:{args.port}")
    try:
        uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
