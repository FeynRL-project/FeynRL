import ast
import builtins
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Entry-point scripts with an `if __name__ == "__main__"` block.
ENTRY_SCRIPTS = ["main_rl.py", "main_sl.py", "main_cl.py"]

def _is_main_guard(node):
    '''
        Return True if *node* is ``if __name__ == "__main__":``.
    '''
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    left, right = test.left, test.comparators[0]
    if (isinstance(left, ast.Name) and left.id == "__name__"
            and isinstance(right, ast.Constant) and right.value == "__main__"):
        return True
    if (isinstance(right, ast.Name) and right.id == "__name__"
            and isinstance(left, ast.Constant) and left.value == "__main__"):
        return True
    return False


def _collect_module_level_names(tree):
    '''
        Names defined at module scope (imports, functions, classes, assignments).
    '''
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
    return names


def _collect_assigned_names(node):
    '''
        All names written (Store context) anywhere inside an AST subtree,
        including function params, exception handler names, and names
        introduced by `import X` / `from X import Y` statements that
        appear inside the subtree (e.g. conditional imports inside an
        `if` block — used by main_rl.py to dispatch to the async/sync
        engine).
    '''
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(child.name)
            for arg in child.args.args + child.args.posonlyargs + child.args.kwonlyargs:
                names.add(arg.arg)
            if child.args.vararg:
                names.add(child.args.vararg.arg)
            if child.args.kwarg:
                names.add(child.args.kwarg.arg)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                names.add(alias.asname or alias.name)
    return names


def _collect_referenced_names(node):
    """All names read (Load context) anywhere inside an AST subtree."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _find_main_block(tree):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If) and _is_main_guard(node):
            return node
    return None


def _get_undefined_names_in_main(filepath):
    '''
        Return set of names referenced in __main__ but never defined.
    '''
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    main_block = _find_main_block(tree)
    if main_block is None:
        return set()

    module_names = _collect_module_level_names(tree)
    main_defined = _collect_assigned_names(main_block)
    main_referenced = _collect_referenced_names(main_block)
    builtin_names = set(dir(builtins))

    available = module_names | main_defined | builtin_names
    return main_referenced - available


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_no_undefined_names_in_main_block(script):
    '''
        Verify that all variable names referenced in ``if __name__ == '__main__'``
        are defined at module scope, locally, or as builtins.
        Uses the ast module to parse each script and verify that every name
        referenced in the `if __name__ == "__main__"` block is defined somewhere
        accessible (module-level, local to the block, or a Python builtin).
    '''
    filepath = os.path.join(REPO_ROOT, script)
    if not os.path.exists(filepath):
        pytest.skip(f"{script} not found")

    undefined = _get_undefined_names_in_main(filepath)
    assert not undefined, (
        f"{script}: undefined names in __main__ block: {sorted(undefined)}. "
        f"This likely indicates a typo or missing import."
    )
