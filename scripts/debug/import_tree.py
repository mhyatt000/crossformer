from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Annotated
import tyro

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "crossformer"


@dataclass
class Config:
    target: Annotated[str, tyro.conf.Positional] = "crossformer.data.grain.loader"
    files: bool = False


def mod_to_paths(mod: str) -> list[Path]:
    base = ROOT.joinpath(*mod.split("."))
    return [base.with_suffix(".py"), base / "__init__.py"]


def resolve_target(target: str) -> Path:
    p = Path(target)
    if p.exists():
        return p.resolve()
    for cand in mod_to_paths(target):
        if cand.exists():
            return cand
    raise FileNotFoundError(target)


def resolve_import(cur: Path, node: ast.AST) -> list[Path]:
    out: list[Path] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if node.level:
            pkg = cur.parent
            for _ in range(node.level - 1):
                pkg = pkg.parent
            base = ".".join(pkg.relative_to(ROOT).parts)
            mod = ".".join([p for p in [base, mod] if p])
        names = [mod]
        names.extend(".".join([p for p in [mod, alias.name] if p]) for alias in node.names if alias.name != "*")
    else:
        return out

    seen: set[Path] = set()
    for name in names:
        if not name.startswith("crossformer"):
            continue
        for cand in mod_to_paths(name):
            if cand.exists() and cand not in seen:
                seen.add(cand)
                out.append(cand)
                break
    return out


def parse_deps(path: Path) -> list[Path]:
    tree = ast.parse(path.read_text())
    out: list[Path] = []
    seen: set[Path] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for dep in resolve_import(path, node):
            if dep in seen or not dep.is_relative_to(PKG):
                continue
            seen.add(dep)
            out.append(dep)
    return out


def build_graph(start: Path) -> dict[Path, list[Path]]:
    graph: dict[Path, list[Path]] = {}
    stack = [start]
    while stack:
        path = stack.pop()
        if path in graph:
            continue
        deps = parse_deps(path)
        graph[path] = deps
        stack.extend(reversed(deps))
    return graph


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def print_tree(
    path: Path,
    graph: dict[Path, list[Path]],
    seen: set[Path] | None = None,
    active: tuple[Path, ...] = (),
    prefix: str = "",
) -> None:
    seen = set() if seen is None else seen
    label = rel(path)
    if path in active:
        print(prefix + label + " [cycle]")
        return
    if path in seen:
        print(prefix + label + " [seen]")
        return
    print(prefix + label)
    seen.add(path)
    deps = graph.get(path, [])
    for i, dep in enumerate(deps):
        branch = "└── " if i == len(deps) - 1 else "├── "
        ext = "    " if i == len(deps) - 1 else "│   "
        print_tree(dep, graph, seen, (*active, path), prefix + branch)


def main(cfg: Config) -> None:
    start = resolve_target(cfg.target)
    graph = build_graph(start)
    print_tree(start, graph)
    if not cfg.files:
        return
    print("\nFILES")
    for path in sorted(graph):
        print(rel(path))


if __name__ == "__main__":
    main(tyro.cli(Config))
