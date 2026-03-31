# Agent Context

Refer to `docs/` for repo specific context
If looking undirected for fixes to make, start with `roadmap/*` for info. remove items from the
roadmap once completed.

# STEERING

## debugging strategy

binary search in code. when tracking a bug or verifying a property, narrow
the search space in half each step — comment out half the pipeline, add an
assert mid-call-chain, or bisect the data. one well-placed check that halves
the remaining candidates is very efficient. also: don't be shy about adding
lots of print statements throughout a call chain — seeing shapes, dtypes,
and values at every stage is a fast way to spot where things go wrong.

## searching code efficiently

prefer grep/rg with -C `n-lines` to get relevant surrounding context with no token waste.
avoid reading entire files when you only need a specific function or block.

## running code and iterative debugging

in general, after making a change, run at least one of [a] an existing script. [b] a new throwaway script or REPL
[c] relevant unit and integration tests. changes are best to be debugged before returning to the human.

this guideline can be ignored IFF the script will incur significant stdout (token debt) in which
case it is better to ask the human if you should run or pipe to a *.log file. similarly can ask for
guidance if the script will take a long time to run (2+min) such as training for a long time.

# SURPRISES

## dof_ids are ground truth for slot ordering

action slots from the grain pipeline are NOT in canonical joint order (j0..j6).
`dof_ids` (from `act.id`) is the ground truth for which DOF is in which slot.
any code that consumes actions by position (rasterization, denormalization by offset,
FK, etc.) must use `dof_ids` to map slots to the correct DOF — never assume `[:7]` = j0..j6.

## uvx tools

uv is first class for environment, but use uvx to install and use ruff, pre-commit, etc

## uv run python

Use `uv run python` instead of bare `python`, even for quick REPL checks. Bare
`python` may use the system interpreter rather than the project environment and
report missing deps incorrectly.

Bad:
```bash
python
python - <<'PY'
from crossformer.utils.spec import spec
PY
```

Good:
```bash
uv run python
uv run python - <<'PY'
from crossformer.utils.spec import spec
PY
```

## tyro booleans

Tyro CLI args use kebab-case flag names like `--batch-size`. Boolean flags are
toggles like `--flag` and `--no-flag`, not `--flag False`.

Bad:
```bash
uv run scripts/train/xflow.py --wandb.use False
uv run scripts/train/xflow.py --myflag True
```

Good:
```bash
uv run scripts/train/xflow.py --wandb.no-use
uv run scripts/train/xflow.py --myflag
```

## RUF003

RUF003 Comment contains ambiguous (MULTIPLICATION SIGN). Did you mean `x` (LATIN SMALL LETTER X)?
use letter x in codebase so this doesnt happen

# TIPS

## findimports

uvx findimports -N -q a/b
use this to parse dependency graph

# BOILERPLATE

# MAIN

# THINGS I LIKE

* I prefer very concise docstrings. and concise code. the purpose of the code
  should be clear from reading its contents

if its a dataclass, then
@dataclass class Myclass:
""" Myclass is a dataclass that represents a simple example. """
    name: str # it is better to describe the argument here
    age: int # and here rather than the docstring

* I prefer OOP paterns for different components.
* wrapper/decorator can sometimes be a good choice
* I also like using a config.create(*args, **kwargs) to create components from
  config objects

* don't be afraid to propose new code restructuring and reorganizing if doing
  so benefits maintainability, but do not do so frivolously

**multi file organization**
*	Use feature-based style rather than layer based. Keeps complexity isolated
*	Add a shared/infra or common/ layer for reusable low-level primitives

### The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!


# THINGS I DON'T LIKE

* excessive nesting. except when necessary, components should be flat. list and
  dict comprehension is fine. jax.tree.map is also fine
* long variable names. this clutters the code. try to keep them short and
  meaningful.
* Long Functions or Methods. multiple responsibilities can be challenging to
  test and maintain. Break down long functions into smaller, single-purpose
  functions to adhere to the Single Responsibility Principle. cyclomatic complexity (CC) should be no more than 8, but 4-5 is better.
* Duplicated Code. keep it dry
