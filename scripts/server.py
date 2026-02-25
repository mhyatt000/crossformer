from __future__ import annotations

from rich import print
import tyro
from webpolicy.server import Server

from crossformer.run.server import Policy, ServerConfig


def main(cfg: ServerConfig):
    print(cfg)
    cfg.policy.verify()

    policy = Policy(cfg.policy)
    server = Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata=None,
    )
    print("serving on", cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
