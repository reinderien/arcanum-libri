#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from types import NoneType
from typing import Optional, Union, Iterable, Any
from subprocess import check_output

from jinja2 import Environment, FileSystemLoader
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
    classes: dict[str, Class],
) -> None:
    env = Environment(
        loader=FileSystemLoader(searchpath='.'),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template('template.html')
    content = template.render(
        branch=get_branch(),
        package=package,
        classes=classes,
        tiers=sort_tiers(classes.values()),
    )

    parent = Path('docs/')
    parent.mkdir(exist_ok=True)
    (parent / 'index.html').write_text(content)


def main() -> None:
    package, classes, = load_data()
    render(package, classes)


if __name__ == '__main__':
    main()
