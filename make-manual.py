#!/usr/bin/env python3

import json
import shutil
import subprocess
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from pprint import pformat
from types import NoneType
from typing import Any, Iterable, Iterator, Optional, Union, NamedTuple, Collection

from jinja2 import Environment, FileSystemLoader

import slimit.ast as ast
from slimit.parser import Parser

parser = Parser()
DATA_ROOT = Path('arcanum')


@dataclass(frozen=True)
class Tier:
    sequence: int
    id: str
    friendly_name: str
    event: Optional['Event']

    @property
    def title(self) -> str:
        if self.event and self.event.name:
            return f'{self.friendly_name}: {self.event.name}'
        return self.friendly_name

    @classmethod
    def from_intro_event(cls, event: 'Event') -> 'Tier':
        if event.id in {'evt_intro', 'evt_scroll', 'evt_alcove'}:
            sequence = -3
            name = 'Apprentice Tier'
        elif event.id == 'evt_helper':
            sequence = -2
            name = 'Job Tier'
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
        for name in ('evt_alcove', 'evt_helper'):
            yield cls.from_intro_event(events[name])

        yield Tier(
            sequence=-1, friendly_name='Neophyte Tier', id='neophyte_pseudotier', event=None,
        )

        for event in events.values():
            if event.id.startswith('tier'):
                yield cls.from_tier_event(event)

    @property
    def tag(self) -> Optional[str]:
        if self.id.startswith('tier'):
            return 't_' + self.id
        if self.id == 'evt_helper':
            return 't_job'
        return None

    @property
    def permalink(self) -> str:
        return f'tier_{self.sequence}'

    @staticmethod
    def group_classes(
        tiers: dict[str, 'Tier'],
        classes: Iterable['Class'],
    ) -> OrderedDict[str, 'Class']:

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


class HasRequirements:
    require: Optional[str] = None
    need: Optional[str] = None

    @staticmethod
    def describe_requirements(source: Union[str, Iterable[str]], index: dict[str, Any]) -> Iterable[str]:
        if isinstance(source, str):
            source = (source,)
        for line in source:
            tree = parser.parse(line)
            mutated = MutatedNode(tree).transform()
            yield from mutated.describe(index)

    def friendly_require(self, index: dict[str, Any]) -> Iterable[str]:
        return self.describe_requirements(self.require, index)

    def friendly_need(self, index: dict[str, Any]) -> Iterable[str]:
        return self.describe_requirements(self.need, index)


class HasRaw:
    raw: dict

    @property
    def formatted_raw(self) -> str:
        return pformat(self.raw)


@dataclass(frozen=True)
class Class(HasRaw, HasRequirements):
    raw: dict
    id: str
    actdesc: Optional[str] = None
    require: Optional[str] = None
    cost: Optional[dict[str, int]] = None
    result: Optional[dict[str, int]] = None
    log: Optional[dict[str, str]] = None
    warn: Optional[bool] = None
    disable: Union[NoneType, str, list[str]] = None
    desc: Optional[str] = None
    actname: Optional[str] = None
    buyname: Optional[str] = None
    need: Optional[str] = None
    alias: Optional[str] = None
    secret: Optional[bool] = None
    flavor: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[str] = None
    mod: Union[NoneType, str, dict[str, Union[bool, float]]] = None
    max: Optional[int] = None

    @property
    def friendly_name(self) -> str:
        return (self.name or self.id).title()

    def get_tier(self, tiers: dict[str, 'Tier']) -> Tier:
        if self.mod is None:
            return tiers['evt_alcove']  # apprentice

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

    def sort_key(self) -> str:
        return self.friendly_name

    @property
    def positive_requirements(self) -> Iterator[str]:
        for source in (self.need, self.require):
            if source:
                tree = parser.parse(source)
                mutated = MutatedNode(tree).transform()
                yield from mutated.requirements()

    @property
    def permalink(self) -> str:
        return f'class_{self.id}'

    @staticmethod
    def load_reverse_deps(classes: dict[str, 'Class']) -> dict[str, list['Class']]:
        requirements = defaultdict(list)
        for dependent_class in classes.values():
            for required_name in dependent_class.positive_requirements:
                depended_class = classes.get(required_name)
                if depended_class is not None:
                    requirements[required_name].append(dependent_class)

        return requirements


@dataclass(frozen=True)
class Event(HasRaw, HasRequirements):
    raw: dict
    id: str
    desc: str
    name: Optional[str] = None
    require: Optional[str] = None
    lock: Optional[Union[str, list[str]]] = None
    disable: Optional[list[str]] = None
    result: Optional[dict[str, float]] = None
    mod: Optional[dict[str, float]] = None

    def get_lock_tiers(self, index: dict[str, Any]) -> list[Tier]:
        if not self.lock:
            return []
        locks = self.lock if isinstance(self.lock, list) else [self.lock]
        results = []
        for lock in locks:
            results.append(index[lock])
        return results


@dataclass(frozen=True)
class Skill(HasRaw, HasRequirements):
    raw: dict
    id: str
    run: dict[str, float]
    mod: dict[str, float]
    need: Optional[Union[str, list[str]]] = None
    desc: Optional[str] = None
    buy: Optional[dict[str, float]] = None
    result: Optional[dict[str, float]] = None
    name: Optional[str] = None
    school: Optional[Union[str, list[str]]] = None
    tags: Optional[str] = None
    verb: Optional[str] = None
    locked: Optional[bool] = None
    require: Optional[str] = None
    level: Optional[int] = None
    flavor: Optional[str] = None
    alias: Optional[str] = None
    effect: Optional[dict[str, float]] = None

    @property
    def friendly_name(self) -> str:
        return (self.name or self.id).title()

    @property
    def permalink(self) -> str:
        return 'skill_' + self.id

    def sort_key(self) -> str:
        return self.friendly_name

    @staticmethod
    def group(
        skills: Collection[dict[str, Any]],
    ) -> tuple[
         dict[str, 'Skill'],
         dict[int, list['Skill']],
     ]:
        by_id = {
            s['id']: Skill(raw=s, **s)
            for s in skills
        }

        by_level = defaultdict(list)
        for skill in by_id.values():
            by_level[skill.level or 0].append(skill)

        for level_skills in by_level.values():
            level_skills.sort(key=Skill.sort_key)

        ordered_by_level = OrderedDict(sorted(by_level.items()))
        return by_id, ordered_by_level


class Database(NamedTuple):
    package: dict[str, Any]
    branch: str
    tiers_by_id: dict[str, Tier]
    classes_by_id: dict[str, Class]
    classes_by_tier: dict[str, Class]
    classes_by_name: list[Class]
    class_deps: dict[str, list[Class]]
    skills_by_level: dict[int, list[Skill]]
    index: dict[str, Union[Tier, Class]]

    @staticmethod
    def _load_json(filename: str) -> dict | list:
        with (DATA_ROOT / filename).with_suffix('.json').open('rb') as f:
            return json.load(f)

    @classmethod
    def _inspect_file(cls, filename: str) -> dict[set[type]]:
        fields = defaultdict(set)
        data = cls._load_json(filename)
        for entry in data:
            for name, value in entry.items():
                fields[name].add(type(value))
        for entry in data:
            for name, types in fields.items():
                if name not in entry:
                    types.add(NoneType)
        return fields

    @classmethod
    def from_json(cls) -> 'Database':
        package = cls._load_json('package')
        print(f'Loaded data for {package["name"]} {package["version"]}')

        classes_by_id = {
            d['id']: Class(raw=d, **d) for d in
            (
                *cls._load_json('data/classes'),
                *cls._load_json('data/hall')['data']['classes']
            )
        }
        events = {e['id']: Event(raw=e, **e) for e in cls._load_json('data/events')}
        tiers = OrderedDict(
            (t.id, t) for t in sorted(
                Tier.from_events(events),
                key=Tier.sort_key,
            )
        )
        tiers_by_tag = {
            ttag: tier
            for tier in tiers.values()
            if (ttag := tier.tag) is not None
        }

        skills_by_id, skills_by_level = Skill.group(cls._load_json('data/skills'))

        index = (
            tiers
            | tiers_by_tag
            | classes_by_id
            | skills_by_id
        )

        return cls(
            package=package,
            branch=cls._get_branch(),
            tiers_by_id=tiers,
            classes_by_id=classes_by_id,
            classes_by_tier=Tier.group_classes(tiers, classes_by_id.values()),
            classes_by_name=sorted(classes_by_id.values(), key=Class.sort_key),
            class_deps=Class.load_reverse_deps(classes_by_id),
            skills_by_level=skills_by_level,
            index=index,
        )

    @staticmethod
    def _get_branch() -> str:
        cmd = shutil.which('git')
        output = subprocess.check_output(
            (cmd, 'branch', '--show-current'),
            cwd=DATA_ROOT, shell=False, text=True,
        )
        return output.rstrip()


def level_name(level: int) -> str:
    if level < 1:
        return 'Base'
    return f'Level {level}'


def render(db: Database) -> None:
    env = Environment(
        loader=FileSystemLoader(searchpath='.'),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template('template.html')
    content = template.render(
        isinstance=isinstance, hasattr=hasattr, len=len, list=list,
        level_name=level_name,
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
