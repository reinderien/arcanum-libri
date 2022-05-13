import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import NoneType
from typing import Optional, Union, Iterator
from urllib.parse import urljoin

from graphviz import Digraph
from requests import Session

import slimit.ast

from slimit.parser import Parser
from slimit.visitors import nodevisitor

parser = Parser()


def key_is_okay(k: str):
    return (
        not k.endswith('rate')
        and not k.endswith('.max')
        and not k.startswith('tier')
        and k not in {
         'research', 'value', 'sp',
        }
    )

@dataclass
class Class:
    id: str
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

    def vars_in(self) -> Iterator[tuple[str, str]]:  # name, full expr
        for source in (self.require, self.need):
            if not source:
                continue
            tree = parser.parse(source)
            for node in nodevisitor.visit(tree):
                if isinstance(node, slimit.ast.Identifier) and node.value != 'g':
                    yield node.value, source

    def vars_out(self) -> Iterator[tuple[str, str]]:
        for dest in (self.mod, self.result):
            if isinstance(dest, str):
                yield dest, dest
            elif isinstance(dest, dict):
                for k, v in dest.items():
                    yield k, f'{k}=={v}'
            elif dest is not None:
                raise NotImplementedError()

    def add_node(self, dot: Digraph):
        dot.node(
            name=self.id,
            label=self.friendly_name,
        )

    def add_edges(self, dot: Digraph, others: dict[str, 'Class']):
        for k, expr in self.vars_in():
            if key_is_okay(k):
                dot.edge(tail_name=k, head_name=self.id, edgetooltip=expr, penwidth='3')
        for k, expr in self.vars_out():
            if key_is_okay(k):
                dot.edge(head_name=k, tail_name=self.id, edgetooltip=expr, penwidth='3')


def download_or_load(
    session: Session,
    cache_dir: Path,
    filename: str,
) -> dict:
    file_path = cache_dir / filename
    inner_dir = file_path.parent
    inner_dir.mkdir(exist_ok=True, parents=True)

    if file_path.is_file():
        with file_path.open() as f:
            doc = json.load(f)
    else:
        urlbase = 'https://gitlab.com/mathiashjelm/arcanum/-/raw/master/'
        url = urljoin(urlbase, filename)
        with session.get(url) as resp:
            resp.raise_for_status()
            file_path.write_bytes(resp.content)
            doc = resp.json()

    return doc


def load_data() -> tuple[dict, list]:
    cache_dir = Path('.cache')

    with Session() as session:
        package = download_or_load(session, cache_dir, 'package.json')
        classes = download_or_load(session, cache_dir, 'data/classes.json')

    return package, classes


def main() -> None:
    package, classes_data = load_data()
    print(f'Loaded data for {package["name"]} {package["version"]}')
    print(f'{len(classes_data)} classes')

    classes = {d['id']: Class(**d) for d in classes_data}

    dot = Digraph(
        comment=f'{package["name"]} {package["version"]}',
        # engine='neato',
        # graph_attr={'scale': '2.0'},
    )

    for class_ in classes.values():
        class_.add_node(dot)
        class_.add_edges(dot, classes)

    dot.render('classes', format='svg')

    return


if __name__ == '__main__':
    main()
