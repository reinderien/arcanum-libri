#!/usr/bin/env python3

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from pprint import pformat
from subprocess import check_output
from types import NoneType
from typing import Any, Iterable, Iterator, Optional, Union, NamedTuple

from jinja2 import Environment, FileSystemLoader

import slimit.ast as ast
from slimit.parser import Parser

parser = Parser()
DATA_ROOT = Path('arcanum')


@dataclass(frozen=True)
class Tier:
    sequence: int
    id: str
    name: str
    event: Optional['Event']

    @classmethod
    def from_intro_event(cls, event: 'Event') -> 'Tier':
        if event.id in {'evt_intro', 'evt_scroll', 'evt_alcove'}:
            sequence = -3
            name = 'Tier: Apprentice'
        elif event.id == 'evt_helper':
            sequence = -2
            name = 'Tier: Job'
        return cls(sequence, event.id, name, event)

    @classmethod
    def from_tier_event(cls, event: 'Event') -> 'Tier':
        sequence = int(event.id.removeprefix('tier'))
        name = f'Tier {sequence}'
        return cls(sequence, event.id, name, event)

    def sort_key(self) -> int:
        return self.sequence

    @classmethod
    def from_events(cls, events: dict[str, 'Event']) -> Iterator['Tier']:
        for name in ('evt_scroll', 'evt_helper'):
            yield cls.from_intro_event(events[name])

        yield Tier(
            sequence=-1, name='Tier: Neophyte', id='neophyte_pseudotier', event=None,
        )

        for event in events.values():
            if event.id.startswith('tier'):
                yield cls.from_tier_event(event)


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

    @property
    def is_positive(self) -> bool:
        if not isinstance(self.node, ast.BinOp):
            return True
        right = self.children[1].node
        if not isinstance(right, ast.Number):
            return True

        op = self.node.op
        n = Decimal(right.value)

        return (
            op == '>'
            or (op in {'>=', '=='} and n > 0)
        )

    def describe_indexed(self, index: dict[str, Any]) -> tuple[Union[str, 'Class', 'Tier'], ...]:
        left, _ = self.children

        if (
            (class_ := index.get(left.node.value))
            and isinstance(class_, Class)
        ):
            if self.is_positive:
                return class_,
            return 'not a ', class_

        left_name = left.node.value
        tier = index.get(left_name)
        if isinstance(tier, Tier):
            if self.is_positive:
                return tier,
            return 'not ', tier

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

    def requirements(self, positive: bool = True) -> Iterator[str]:
        if isinstance(self.node, ast.Identifier):
            if positive:
                yield self.node.value

        this_positive = self.is_positive
        for child in self.children:
            yield from child.requirements(
                positive=positive ^ (not this_positive)
            )


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

    def get_tier(self, tiers: dict[str, 'Tier']) -> Tier:
        if self.mod is None:
            return tiers['evt_scroll']  # apprentice

        if isinstance(self.mod, str):
            return tiers[self.mod]  # tier 0

        for k, v in self.mod.items():
            if k.startswith('tier'):
                return tiers[k]

        if self.result.get('evt_helper'):
            return tiers['evt_helper']  # job

        return tiers['neophyte_pseudotier']

    @property
    def modifier_map(self) -> dict[str, Any]:
        if not self.mod:
            return {}
        if isinstance(self.mod, str):
            return {self.mod: True}
        return self.mod

    @property
    def disabled_actions(self) -> tuple[str, ...]:
        if not self.disable:
            return ()
        if isinstance(self.disable, str):
            return self.disable,
        return tuple(self.disable)

    @staticmethod
    def describe_requirements(source: str, index: dict[str, Any]) -> Iterable[str]:
        tree = parser.parse(source)
        mutated = MutatedNode(tree).transform()
        return mutated.describe(index)

    def friendly_require(self, index: dict[str, Any]) -> Iterable[str]:
        return self.describe_requirements(self.require, index)

    def friendly_need(self, index: dict[str, Any]) -> Iterable[str]:
        return self.describe_requirements(self.need, index)

    def sort_key(self) -> str:
        return self.friendly_name

    @property
    def positive_requirements(self) -> Iterator[str]:
        for source in (self.need, self.require):
            if source:
                tree = parser.parse(source)
                mutated = MutatedNode(tree).transform()
                yield from mutated.requirements()


@dataclass(frozen=True)
class Event:
    id: str
    desc: str
    name: Optional[str] = None
    require: Optional[str] = None
    lock: Optional[str] = None
    disable: Optional[list[str]] = None
    result: Optional[dict[str, float]] = None
    mod: Optional[dict[str, float]] = None


class Database(NamedTuple):
    package: dict[str, Any]
    branch: str
    tiers_by_id: dict[str, Tier]
    classes_by_id: dict[str, Class]
    classes_by_tier: dict[str, Class]
    class_deps: dict[str, list[Class]]
    index: dict[str, Union[Tier, Class]]

    @staticmethod
    def _load_json(filename: str) -> dict | list:
        with (DATA_ROOT / filename).with_suffix('.json').open() as f:
            return json.load(f)

    @classmethod
    def from_json(cls) -> 'Database':
        package = cls._load_json('package')
        print(f'Loaded data for {package["name"]} {package["version"]}')

        classes = {d['id']: Class(raw=d, **d) for d in cls._load_json('data/classes')}
        events = {e['id']: Event(**e) for e in cls._load_json('data/events')}
        tiers = OrderedDict(
            (t.id, t) for t in sorted(
                Tier.from_events(events),
                key=Tier.sort_key,
            )
        )

        index = tiers | classes

        return cls(
            package=package,
            branch=cls._get_branch(),
            tiers_by_id=tiers,
            classes_by_id=classes,
            classes_by_tier=cls._group_classes(tiers, classes.values()),
            class_deps=cls._load_class_reverse_deps(classes),
            index=index,
        )

    @staticmethod
    def _group_classes(
        tiers: dict[str, Tier],
        classes: Iterable[Class],
    ) -> OrderedDict[str, Class]:

        classes_by_tier = defaultdict(list)
        for class_ in classes:
            tier = class_.get_tier(tiers)
            classes_by_tier[tier.id].append(class_)
        for group in classes_by_tier.values():
            group.sort(key=Class.sort_key)

        def sort_key(pair):
            tier_id, class_ = pair
            return tiers[tier_id].sequence

        return OrderedDict(sorted(
            classes_by_tier.items(),
            key=sort_key,
        ))

    @staticmethod
    def _load_class_reverse_deps(classes: dict[str, Class]) -> dict[str, list[Class]]:
        requirements = defaultdict(list)
        for dependent_class in classes.values():
            for required_name in dependent_class.positive_requirements:
                depended_class = classes.get(required_name)
                if depended_class is not None:
                    requirements[required_name].append(dependent_class)

        return requirements

    @staticmethod
    def _get_branch() -> str:
        output = check_output(
            ('git', 'branch', '--show-current'),
            cwd=DATA_ROOT, shell=True, text=True,
        )
        return output.rstrip()


def render(db: Database) -> None:
    env = Environment(
        loader=FileSystemLoader(searchpath='.'),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template('template.html')
    content = template.render(
        isinstance=isinstance, len=len,
        Tier=Tier, Class=Class,
        **db._asdict(),
    )

    parent = Path('docs/')
    parent.mkdir(exist_ok=True)
    (parent / 'index.html').write_text(content, encoding='utf-8')


def main() -> None:
    db = Database.from_json()
    render(db)


if __name__ == '__main__':
    main()
