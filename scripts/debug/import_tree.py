from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Annotated
import tyro

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "crossformer"


@dataclass
class Config:
    target: Annotated[tuple[str, ...], tyro.conf.Positional]
    """One or more files/modules; their trees are printed as a deduplicated union."""
    files: bool = False
    """Append the flat closure of reached files."""
    depth: int | None = None
    """Limit tree depth (counts file hops)."""
    funcs: bool = False
    """Function-level use graph: expand each file into its scopes (functions/methods/
    <module>) and show which crossformer modules each scope references."""
    outlier: tuple[str, ...] = ()
    """Watch group: files, dirs, or 'file:func' specs. Prints the tree of any watched
    item NOT reached by the target(s) — i.e. dead relative to those entry points."""
    dead: bool = False
    """Report top-level functions/classes in the group (--outlier paths, else the
    targets) that are referenced by nothing in the repo. Uses a symbol-reference
    graph (intra-file calls + imported-symbol calls). 'file:func' targets are forced
    live. Cannot see dynamic refs (ModuleSpec strings, registries, getattr, __all__)."""


# --------------------------------------------------------------------------- #
# resolution helpers
# --------------------------------------------------------------------------- #
def mod_to_paths(mod: str) -> list[Path]:
    base = ROOT.joinpath(*mod.split("."))
    return [base.with_suffix(".py"), base / "__init__.py"]


def first_existing(mod: str) -> Path | None:
    for cand in mod_to_paths(mod):
        if cand.exists() and cand.is_relative_to(PKG):
            return cand
    return None


def resolve_target(target: str) -> Path:
    p = Path(target)
    if p.exists():
        return p.resolve()
    for cand in mod_to_paths(target):
        if cand.exists():
            return cand
    raise FileNotFoundError(target)


def safe_parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except (OSError, SyntaxError, ValueError):
        return None


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- #
# file-level graph (default mode) — unchanged behaviour
# --------------------------------------------------------------------------- #
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
    tree = safe_parse(path)
    if tree is None:
        return []
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


def build_graph(start: Path, graph: dict[Path, list[Path]] | None = None) -> dict[Path, list[Path]]:
    graph = {} if graph is None else graph
    stack = [start]
    while stack:
        path = stack.pop()
        if path in graph:
            continue
        deps = parse_deps(path)
        graph[path] = deps
        stack.extend(reversed(deps))
    return graph


def print_tree(
    path: Path,
    graph: dict[Path, list[Path]],
    seen: set[Path] | None = None,
    active: tuple[Path, ...] = (),
    prefix: str = "",
    depth: int | None = None,
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
    if depth == 0:
        return
    deps = graph.get(path, [])
    next_depth = None if depth is None else depth - 1
    for i, dep in enumerate(deps):
        branch = "└── " if i == len(deps) - 1 else "├── "
        print_tree(dep, graph, seen, (*active, path), prefix + branch, next_depth)


# --------------------------------------------------------------------------- #
# function-level use graph (--funcs)
# --------------------------------------------------------------------------- #
def attr_to_str(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def import_bindings(cur: Path, tree: ast.Module) -> tuple[dict[str, Path], list[tuple[str, Path]]]:
    """Map bound local names → the crossformer module file they resolve to.

    `simple` covers `import x as y` and `from x import y`; `dotted` covers
    `import crossformer.a.b` (referenced via the full dotted path).
    """
    simple: dict[str, Path] = {}
    dotted: list[tuple[str, Path]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("crossformer"):
                    continue
                path = first_existing(alias.name)
                if path is None:
                    continue
                if alias.asname:
                    simple[alias.asname] = path
                else:
                    dotted.append((alias.name, path))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                pkg = cur.parent
                for _ in range(node.level - 1):
                    pkg = pkg.parent
                try:
                    base = ".".join(pkg.relative_to(ROOT).parts)
                except ValueError:
                    base = ""
                mod = ".".join(p for p in [base, mod] if p)
            if not mod.startswith("crossformer"):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                path = first_existing(f"{mod}.{alias.name}") or first_existing(mod)
                if path is None:
                    continue
                simple[alias.asname or alias.name] = path
    return simple, dotted


def extract_scope_nodes(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """Split a module into named scopes → the AST nodes whose bodies define that scope.

    Scopes: each top-level function, each `Class.method`, each class body (bases +
    class-level statements), and `<module>` for everything else at top level.
    """
    scopes: dict[str, list[ast.AST]] = {}
    module_glue: list[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes[stmt.name] = [stmt]
        elif isinstance(stmt, ast.ClassDef):
            glue: list[ast.AST] = [*stmt.decorator_list, *stmt.bases, *(kw.value for kw in stmt.keywords)]
            for member in stmt.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scopes[f"{stmt.name}.{member.name}"] = [member]
                else:
                    glue.append(member)
            scopes[stmt.name] = glue
        else:
            module_glue.append(stmt)
    scopes["<module>"] = module_glue
    return scopes


def resolve_scope_deps(nodes: list[ast.AST], simple: dict[str, Path], dotted: list[tuple[str, Path]]) -> list[Path]:
    names: set[str] = set()
    attrs: list[str] = []
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                s = attr_to_str(n)
                if s:
                    attrs.append(s)
    deps: set[Path] = set()
    for nm in names:
        if nm in simple:
            deps.add(simple[nm])
    for a in attrs:
        for dot, path in dotted:
            if a == dot or a.startswith(dot + "."):
                deps.add(path)
    return sorted(deps, key=rel)


_scope_cache: dict[Path, dict[str, list[Path]]] = {}


def scopes_of(path: Path) -> dict[str, list[Path]]:
    """Return {scope_name: [crossformer module deps]} for depful scopes of `path`.

    Imports that no scope references are attributed to `<module>`, so the union of
    scope deps equals the file-level closure (function mode reaches the same files).
    """
    if path in _scope_cache:
        return _scope_cache[path]
    tree = safe_parse(path)
    if tree is None:
        _scope_cache[path] = {}
        return {}
    simple, dotted = import_bindings(path, tree)
    result: dict[str, list[Path]] = {}
    used: set[Path] = set()
    for name, nodes in extract_scope_nodes(tree).items():
        deps = resolve_scope_deps(nodes, simple, dotted)
        if deps:
            result[name] = deps
            used.update(deps)
    leftover = sorted((set(simple.values()) | {p for _, p in dotted}) - used, key=rel)
    if leftover:
        result["<module>"] = sorted(set(result.get("<module>", [])) | set(leftover), key=rel)
    _scope_cache[path] = result
    return result


def build_scope_graph(starts: set[Path]) -> dict[Path, dict[str, list[Path]]]:
    graph: dict[Path, dict[str, list[Path]]] = {}
    stack = list(starts)
    while stack:
        path = stack.pop()
        if path in graph:
            continue
        scopes = scopes_of(path)
        graph[path] = scopes
        for deps in scopes.values():
            stack.extend(d for d in deps if d not in graph)
    return graph


def reach_scopes(targets: list[Path], graph: dict[Path, dict[str, list[Path]]]) -> tuple[set[Path], set[tuple[Path, str]]]:
    files: set[Path] = set()
    scopes: set[tuple[Path, str]] = set()
    stack = list(targets)
    while stack:
        path = stack.pop()
        if path in files:
            continue
        files.add(path)
        for name, deps in graph.get(path, {}).items():
            scopes.add((path, name))
            stack.extend(deps)
    return files, scopes


def scope_label(name: str) -> str:
    return name if name == "<module>" else name + "()"


def print_scope_fn(
    deps: list[Path],
    name: str,
    graph: dict[Path, dict[str, list[Path]]],
    seen: set[Path],
    active: tuple[Path, ...],
    prefix: str,
    depth: int | None,
) -> None:
    print(prefix + scope_label(name))
    next_depth = None if depth is None else depth - 1
    for i, dep in enumerate(deps):
        branch = "└── " if i == len(deps) - 1 else "├── "
        print_file_fn(dep, graph, seen, active, prefix + branch, next_depth)


def print_file_fn(
    path: Path,
    graph: dict[Path, dict[str, list[Path]]],
    seen: set[Path],
    active: tuple[Path, ...] = (),
    prefix: str = "",
    depth: int | None = None,
) -> None:
    label = rel(path)
    if path in active:
        print(prefix + label + " [cycle]")
        return
    if path in seen:
        print(prefix + label + " [seen]")
        return
    print(prefix + label)
    seen.add(path)
    if depth == 0:
        return
    scopes = [(n, d) for n, d in graph.get(path, {}).items() if d]
    for i, (name, deps) in enumerate(scopes):
        branch = "└── " if i == len(scopes) - 1 else "├── "
        print_scope_fn(deps, name, graph, seen, (*active, path), prefix + branch, depth)


# --------------------------------------------------------------------------- #
# outlier watch group (--outlier)
# --------------------------------------------------------------------------- #
@dataclass
class Outlier:
    kind: str  # "file" | "dir" | "func"
    path: Path
    func: str | None = None
    roots: set[Path] = field(default_factory=set)


def parse_outliers(specs: tuple[str, ...]) -> list[Outlier]:
    out: list[Outlier] = []
    for spec in specs:
        if ":" in spec:
            fpart, fn = spec.rsplit(":", 1)
            path = resolve_target(fpart)
            out.append(Outlier("func", path, fn, {path}))
            continue
        p = Path(spec)
        if p.exists() and p.is_dir():
            files = {f.resolve() for f in p.rglob("*.py")}
            out.append(Outlier("dir", p.resolve(), None, files))
        else:
            path = resolve_target(spec)
            out.append(Outlier("file", path, None, {path}))
    return out


def print_outliers(
    outliers: list[Outlier],
    graph: dict[Path, dict[str, list[Path]]],
    reached_files: set[Path],
    reached_scopes: set[tuple[Path, str]],
    funcs: bool,
    depth: int | None,
) -> None:
    print("\nOUTLIERS (trees shown only for items NOT reached above)")
    for o in outliers:
        if o.kind == "func":
            func = o.func or ""
            if not funcs:
                print(f"  {rel(o.path)}:{func} — function granularity needs --funcs; checking file")
                if o.path not in reached_files:
                    seen: set[Path] = set()
                    print(f"\n[unreached] {rel(o.path)}")
                    print_file_fn(o.path, graph, seen, (), "", depth)
                continue
            if (o.path, func) in reached_scopes:
                print(f"  [reached] {rel(o.path)}:{func}")
                continue
            deps = graph.get(o.path, {}).get(func)
            print(f"\n[unreached] {rel(o.path)}:{func}")
            if not deps:
                print("  (no such scope, or scope has no crossformer deps)")
            else:
                print_scope_fn(deps, func, graph, {o.path}, (o.path,), "", depth)
        else:
            targets = sorted(o.roots, key=rel) if o.kind == "dir" else [o.path]
            for path in targets:
                if path in reached_files:
                    print(f"  [reached] {rel(path)}")
                    continue
                seen = set()
                print(f"\n[unreached] {rel(path)}")
                if funcs:
                    print_file_fn(path, graph, seen, (), "", depth)
                else:
                    print_tree(path, build_graph(path), seen=seen, depth=depth)


# --------------------------------------------------------------------------- #
# dead-function detection (--dead): symbol-reference graph over the whole repo
# --------------------------------------------------------------------------- #
_tls_cache: dict[Path, dict[str, str]] = {}
_methods_cache: dict[Path, dict[str, list[str]]] = {}


def top_symbols(path: Path) -> dict[str, str]:
    """{name: 'func'|'class'} for top-level defs/classes; also fills method cache."""
    if path in _tls_cache:
        return _tls_cache[path]
    tree = safe_parse(path)
    syms: dict[str, str] = {}
    methods: dict[str, list[str]] = {}
    if tree is not None:
        for s in tree.body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                syms[s.name] = "func"
            elif isinstance(s, ast.ClassDef):
                syms[s.name] = "class"
                methods[s.name] = [m.name for m in s.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    _tls_cache[path] = syms
    _methods_cache[path] = methods
    return syms


def symbol_bindings(cur: Path, tree: ast.Module) -> tuple[dict[str, Path], list[tuple[str, Path]], dict[str, tuple[Path, str]]]:
    """Like import_bindings, plus `sym`: local name → (module_path, symbol) for
    `from mod import symbol` where symbol is a member (not a submodule) of mod."""
    mod: dict[str, Path] = {}
    dotted: list[tuple[str, Path]] = []
    sym: dict[str, tuple[Path, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("crossformer"):
                    continue
                path = first_existing(alias.name)
                if path is None:
                    continue
                if alias.asname:
                    mod[alias.asname] = path
                else:
                    dotted.append((alias.name, path))
        elif isinstance(node, ast.ImportFrom):
            modstr = node.module or ""
            if node.level:
                pkg = cur.parent
                for _ in range(node.level - 1):
                    pkg = pkg.parent
                try:
                    base = ".".join(pkg.relative_to(ROOT).parts)
                except ValueError:
                    base = ""
                modstr = ".".join(p for p in [base, modstr] if p)
            if not modstr.startswith("crossformer"):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                sub = first_existing(f"{modstr}.{alias.name}")
                bound = alias.asname or alias.name
                if sub is not None:
                    mod[bound] = sub
                else:
                    base_path = first_existing(modstr)
                    if base_path is None:
                        continue
                    mod[bound] = base_path
                    sym[bound] = (base_path, alias.name)
    return mod, dotted, sym


def scope_symbol_refs(path: Path) -> dict[str, set[tuple[Path, str]]]:
    """{scope_name: {(module_path, symbol) it references}} — intra-file + cross-file."""
    tree = safe_parse(path)
    if tree is None:
        return {}
    mod_bind, dotted, sym = symbol_bindings(path, tree)
    locals_ = top_symbols(path)
    out: dict[str, set[tuple[Path, str]]] = {}
    for name, nodes in extract_scope_nodes(tree).items():
        names: set[str] = set()
        attrs: list[str] = []
        for node in nodes:
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.Attribute):
                    s = attr_to_str(n)
                    if s:
                        attrs.append(s)
        tgts: set[tuple[Path, str]] = set()
        for nm in names:
            if nm in locals_:
                tgts.add((path, nm))
            if nm in sym:
                tgts.add(sym[nm])
        for a in attrs:
            root, _, rest = a.partition(".")
            if rest and root in mod_bind and root not in sym:
                tgts.add((mod_bind[root], rest.split(".")[0]))
            for dot, dp in dotted:
                if a.startswith(dot + "."):
                    tgts.add((dp, a[len(dot) + 1 :].split(".")[0]))
        out[name] = tgts
    return out


_resolve_cache: dict[tuple[Path, str], tuple[Path, str] | None] = {}


def resolve_symbol(tp: Path, ts: str, _seen: frozenset[tuple[Path, str]] = frozenset()) -> tuple[Path, str] | None:
    """Follow re-exports: if `ts` isn't defined in `tp` but `tp` (e.g. an __init__)
    imports it from a submodule, chase to where it's actually defined."""
    key = (tp, ts)
    if key in _resolve_cache:
        return _resolve_cache[key]
    if ts in top_symbols(tp):
        _resolve_cache[key] = key
        return key
    result: tuple[Path, str] | None = None
    tree = safe_parse(tp)
    if tree is not None and key not in _seen:
        _, _, sym = symbol_bindings(tp, tree)
        if ts in sym:
            np, ns = sym[ts]
            result = resolve_symbol(np, ns, _seen | {key})
    _resolve_cache[key] = result
    return result


def all_py_files() -> list[Path]:
    out: list[Path] = []
    for sub in ("crossformer", "scripts", "tests", "config", "wip"):
        d = ROOT / sub
        if d.exists():
            out.extend(d.rglob("*.py"))
    out.extend(ROOT.glob("*.py"))
    skip = {".venv", "site-packages", ".git", "node_modules", "__pycache__"}
    return [p for p in out if skip.isdisjoint(p.parts)]


def parse_entry(spec: str) -> tuple[Path, str | None]:
    if ":" in spec:
        fpart, func = spec.rsplit(":", 1)
        return resolve_target(fpart), func
    return resolve_target(spec), None


def find_dead(entries: list[tuple[Path, str | None]], group_files: set[Path]) -> dict[tuple[Path, str], str]:
    """Return {(path, symbol): kind} for group top-level funcs/classes referenced by
    nothing live. `entries` with a func are forced live (declared entrypoints)."""
    referrers: dict[tuple[Path, str], set[tuple[Path, str]]] = {}
    for p in all_py_files():
        for scope_name, tgts in scope_symbol_refs(p).items():
            sid = (p, scope_name)
            for tp, ts in tgts:
                canon = resolve_symbol(tp, ts)
                if canon is None:
                    referrers.setdefault((tp, "<module>"), set()).add(sid)
                    continue
                cp, cs = canon
                referrers.setdefault(canon, set()).add(sid)
                if top_symbols(cp).get(cs) == "class":
                    for m in _methods_cache.get(cp, {}).get(cs, []):
                        referrers.setdefault((cp, f"{cs}.{m}"), set()).add(sid)

    candidates: dict[tuple[Path, str], str] = {}
    for p in group_files:
        for s, k in top_symbols(p).items():
            candidates[(p, s)] = k

    live: set[tuple[Path, str]] = {(p, f) for p, f in entries if f is not None}
    changed = True
    while changed:
        changed = False
        for c in candidates:
            if c in live:
                continue
            for sid in referrers.get(c, ()):
                if sid not in candidates or sid in live:  # referred to by a live scope
                    live.add(c)
                    changed = True
                    break
    return {c: k for c, k in candidates.items() if c not in live}


def report_dead(entries: list[tuple[Path, str | None]], group_files: set[Path]) -> None:
    dead = find_dead(entries, group_files)
    print("DEAD (top-level funcs/classes in group with no live referrer)")
    if not dead:
        print("  (none)")
        return
    for path in sorted({p for p, _ in dead}, key=rel):
        syms = sorted(s for (p, s), _ in dead.items() if p == path)
        print(f"\n{rel(path)}")
        for s in syms:
            print(f"  {dead[(path, s)]:5} {s}")


# --------------------------------------------------------------------------- #
def main(cfg: Config) -> None:
    if cfg.dead:
        entries = [parse_entry(t) for t in cfg.target]
        group = set()
        for o in parse_outliers(cfg.outlier):
            group |= o.roots
        if not group:
            group = {p for p, _ in entries}
        report_dead(entries, group)
        return

    targets = [resolve_target(t) for t in cfg.target]
    outliers = parse_outliers(cfg.outlier)

    if cfg.funcs:
        roots = set(targets)
        for o in outliers:
            roots |= o.roots
        graph = build_scope_graph(roots)
        reached_files, reached_scopes = reach_scopes(targets, graph)
        seen: set[Path] = set()
        for t in targets:
            print_file_fn(t, graph, seen, (), "", cfg.depth)
        if cfg.files:
            print("\nFILES")
            for path in sorted(reached_files, key=rel):
                print(rel(path))
        if outliers:
            print_outliers(outliers, graph, reached_files, reached_scopes, True, cfg.depth)
        return

    fgraph: dict[Path, list[Path]] = {}
    for t in targets:
        build_graph(t, fgraph)
    seen = set()
    for t in targets:
        print_tree(t, fgraph, seen=seen, depth=cfg.depth)
    if cfg.files:
        print("\nFILES")
        for path in sorted(fgraph, key=rel):
            print(rel(path))
    if outliers:
        print_outliers(outliers, {}, set(fgraph), set(), False, cfg.depth)


if __name__ == "__main__":
    main(tyro.cli(Config))
