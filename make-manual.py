#!/usr/bin/env python3

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from pprint import pformat
from types import NoneType
from typing import Optional, Union, Iterable, Any, Iterator
from subprocess import check_output

from jinja2 import Environment, FileSystemLoader

import slimit.ast as ast
from slimit.parser import Parser

parser = Parser()
DATA_ROOT = Path('arcanum')


@dataclass(frozen=True)
class Tier:
    sequence: int
    name: str

    @classmethod
    def from_name(cls, name: str) -> 'Tier':
        tier_sequence = {
            'apprentice': -3,
            'job': -2,
            'neophyte': -1,
        }
        return cls(
            sequence=tier_sequence[name],
            name=f'Tier: {name.title()}',
        )

    @classmethod
    def from_tier_str(cls, tier_str: str) -> 'Tier':
        sequence = int(tier_str.removeprefix('tier'))
        return cls(
            sequence=sequence,
            name=f'Tier {sequence}',
        )

    def sort_key(self) -> int:
        return self.sequence


class MutatedNode:
    def __init__(self, node: ast.Node) -> None:
        self.node = node
        self.children: list[MutatedNode] = [MutatedNode(c) for c in node.children()]

    def __str__(self):
        s = (
            type(self.node).__name__
            + ' ' + getattr(self.node, 'op', '')
            + ' ' + getattr(self.node, 'value', '')
        )
        return s

    def __repr__(self):
        return str(self)

    def transform(self) -> Optional['MutatedNode']:
        new_root = self

        # Drop outer Program and ExprStatement wrappers
        if isinstance(new_root.node, ast.Program):
            new_root, = new_root.children
        if isinstance(new_root.node, ast.ExprStatement):
            new_root, = new_root.children

        if isinstance(new_root.node, ast.Identifier) and new_root.node.value in {'g', 'value'}:
            return None

        # Flatten nested binary operators that match.
        if isinstance(new_root.node, ast.BinOp) and new_root.node.op in {'+', '||', '&&'}:
            i = len(new_root.children)-1
            while i >= 0:
                child = new_root.children[i]  # left or right of outer
                if isinstance(child.node, ast.BinOp) and child.node.op == new_root.node.op:
                    new_root.children.extend(child.children)
                    new_root.children.pop(i)
                    i += len(child.children)
                i -= 1

        new_children = []
        for child in new_root.children:
            replacement = child.transform()
            if replacement is not None:
                new_children.append(replacement)
        new_root.children = new_children

        if len(new_children) == 1:
            if isinstance(self.node, ast.DotAccessor):
                return new_children[0]
            else:
                print('Warning: singleton', str(self))

        return new_root

    def print(self, level=0) -> None:
        print('    '*level + str(self))
        for child in self.children:
            child.print(level=level+1)

    def describe(self, index: dict[str, Any]) -> Iterator[Union[str, 'Class']]:
        if isinstance(self.node, (ast.Number, ast.Identifier)):
            yield index.get(self.node.value, self.node.value)
            return

        if isinstance(self.node, ast.BinOp):
            if (
                len(self.children) == 2
                and isinstance(class_node := self.children[0].node, ast.Identifier)
                and (class_ := index.get(class_node.value))
                and isinstance(class_, Class)
                and isinstance(num_node := self.children[1].node, ast.Number)
            ):
                op = self.node.op
                n = Decimal(num_node.value)
                positive = (
                    op == '>'
                    or (op in {'>=', '=='} and n > 0)
                )
                if not positive:
                    yield 'not a '
                yield class_
                return

            seps = {
                '||': ' or ',
                '&&': ', ',
                '+': ' or ',
                '>': ' greater than ',
                '>=': ' at least ',
                '<': ' less than ',
                '<=': ' at most ',
                '==': ' of ',
            }
            sep = seps.get(self.node.op)
            if sep is None:
                raise ValueError(f'Operator {self.node.op} is not supported')
            yield from self.children[0].describe(index)
            for child in self.children[1:]:
                yield sep
                yield from child.describe(index)

        elif isinstance(self.node, ast.DotAccessor):
            for child in self.children:
                yield ' '
                yield from child.describe(index)

        else:
            raise ValueError(f'Not supported: {self}')


@dataclass(frozen=True)
class Class:
    id: str
    raw: dict
    actdesc: Optional[str] = None
    require: Optional[str] = None
    cost: Optional[dict[str, int]] = None
    result: Optional[dict[str, int]] = None
    log: Optional[dict[str, str]] = None
    warn: Optional[bool] = None
    disable: Optional[list[str]] = None
    desc: Optional[str] = None
    actname: Optional[str] = None
    buyname: Optional[str] = None
    need: Optional[str] = None
    alias: Optional[str] = None
    secret: Optional[bool] = None
    flavor: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[str] = None
    mod: Union[str, NoneType, dict[str, Union[bool, float]]] = None

    @property
    def friendly_name(self) -> str:
        return (self.name or self.id).title()

    @property
    def formatted_raw(self) -> str:
        return pformat(self.raw)

    @property
    def tier(self) -> Tier:
        if self.mod is None:
            return Tier.from_name(self.id)

        if isinstance(self.mod, str):
            return Tier.from_tier_str(self.mod)

        for k, v in self.mod.items():
            if k.startswith('tier'):
                return Tier.from_tier_str(k)

        if self.result.get('evt_helper'):
            return Tier.from_name('job')

        return Tier.from_name(self.id)

    @property
    def modifier_map(self) -> dict[str, Any]:
        if not self.mod:
            return {}
        if isinstance(self.mod, str):
            return {self.mod: True}
        return self.mod

    @property
    def disabled_actions(self) -> tuple[str]:
        if not self.disable:
            return ()
        if isinstance(self.disable, str):
            return self.disable,
        return tuple(self.disable)

    @staticmethod
    def parse_requirements(source: str, index: dict[str, Any]) -> Iterable[str]:
        tree = parser.parse(source)
        mutated = MutatedNode(tree).transform()
        return mutated.describe(index)

    def friendly_require(self, index: dict[str, Any]) -> Iterable[str]:
        return self.parse_requirements(self.require, index)

    def friendly_need(self, index: dict[str, Any]) -> Iterable[str]:
        return self.parse_requirements(self.need, index)


def sort_tiers(classes: Iterable[Class]) -> list[str]:
    unsorted = {c.tier for c in classes}
    return sorted(unsorted, key=Tier.sort_key)


def get_branch() -> str:
    output = check_output(
        ('/usr/bin/git', 'branch', '--show-current'),
        cwd=DATA_ROOT, shell=False, text=True,
    )
    return output.rstrip()


def load_json(filename: str) -> dict | list:

    with (DATA_ROOT / filename).with_suffix('.json').open() as f:
        return json.load(f)


def load_data() -> tuple[
    dict[str, Any],    # package metadata
    dict[str, Class],  # classes
]:
    package = load_json('package')
    print(f'Loaded data for {package["name"]} {package["version"]}')

    classes_json = load_json('data/classes')
    classes = {d['id']: Class(raw=d, **d) for d in classes_json}
    print(f'{len(classes)} classes')

    return package, classes,


def render(
    package: dict,
    index: dict[str, Any],
    classes: dict[str, Class],
) -> None:
    env = Environment(
        loader=FileSystemLoader(searchpath='.'),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template('template.html')
    content = template.render(
        isinstance=isinstance,
        Class=Class,
        branch=get_branch(),
        package=package,
        index=index,
        classes=classes,
        tiers=sort_tiers(classes.values()),
    )

    parent = Path('docs/')
    parent.mkdir(exist_ok=True)
    (parent / 'index.html').write_text(content)


def main() -> None:
    package, classes, = load_data()
    index = classes  # will be expanded
    render(package, index, classes)


if __name__ == '__main__':
    main()
