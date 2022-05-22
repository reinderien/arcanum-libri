#!/usr/bin/env python3

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from pprint import pformat
from subprocess import check_output
from types import NoneType
from typing import Any, Iterable, Iterator, Optional, Union

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
        if name == 'evt_helper':
            name = 'job'
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

    @staticmethod
    def sort_key(pair: tuple['Tier', Any]) -> int:
        tier, by_tier = pair
        return tier.sequence


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

    def __repr__(self) -> str:
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

    def describe_indexed(self, index: dict[str, Any]) -> tuple[Union[str, 'Class', 'Tier'], ...]:
        left, right = self.children
        op = self.node.op
        n = Decimal(right.node.value)
        positive = (
            op == '>'
            or (op in {'>=', '=='} and n > 0)
        )

        if (
            (class_ := index.get(left.node.value))
            and isinstance(class_, Class)
        ):
            if not positive:
                return 'not a ', class_
            return class_,

        tier = None
        left_name = left.node.value
        if left_name.startswith('tier'):
            tier = Tier.from_tier_str(left_name)
        elif left_name == 'evt_helper':
            tier = Tier.from_name(left_name)
        if tier:
            if not positive:
                return 'not ', tier
            return tier,

        return ()

    def describe_binop(self, index: dict[str, Any]) -> Iterator[Union[str, 'Class', 'Tier']]:
        left, right = self.children
        if (
            isinstance(left.node, ast.Identifier) and
            isinstance(right.node, ast.Number)
        ):
            fragments = self.describe_indexed(index)
            if fragments:
                yield from fragments
                return

        seps = {
            '||': ' or ',
            '&&': ', ',
            '+': ' or ',
            '>': ' more than ',
            '>=': ' at least ',
            '<': ' less than ',
            '<=': ' at most ',
            '==': ' of ',
        }
        yield from left.describe(index)
        yield seps[self.node.op]
        yield from right.describe(index)

    def describe(self, index: dict[str, Any]) -> Iterator[Union[str, 'Class', 'Tier']]:
        if isinstance(self.node, ast.BinOp):
            yield from self.describe_binop(index)
            return
        if isinstance(self.node, (ast.Number, ast.Identifier)):
            yield index.get(self.node.value, self.node.value)
        for child in self.children:
            yield ' '
            yield from child.describe(index)


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
            return Tier.from_name('evt_helper')

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

    def sort_key(self) -> str:
        return self.friendly_name


def sort_tiers(classes: Iterable[Class]) -> OrderedDict:
    classes_by_tier = defaultdict(list)
    for class_ in classes:
        classes_by_tier[class_.tier].append(class_)
    for group in classes_by_tier.values():
        group.sort(key=Class.sort_key)

    return OrderedDict(sorted(
        classes_by_tier.items(),
        key=Tier.sort_key,
    ))


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
        len=len,
        Tier=Tier,
        Class=Class,
        branch=get_branch(),
        package=package,
        index=index,
        classes_by_tier=sort_tiers(classes.values()),
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
