import json
from dataclasses import dataclass


@dataclass
class Meta:
    all: list[str]
    target: list[str]
    tool: list[str]
    static: list[str]
    dist: list[str]
    embed: list[str]
    multicat: list[str]
    best_v1: list[str]
    intersect_v1: list[str]


def read_meta():
    with open('../data/columns.json', 'r') as fp:
        meta = json.load(fp)
    return Meta(
        all=meta['all'],
        target=meta['target'],
        tool=meta['tool'],
        static=meta['static'],
        dist=meta['dist'],
        embed=meta['embed'],
        multicat=meta['multicat'],
        best_v1=meta['best-v1'],
        intersect_v1=meta['intersect-v1'],
    )
