from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Union

@dataclass(frozen=True, order=True)
class Variable:
    name: str
    def __str__(self) -> str:
        return f"?{self.name}"

@dataclass(frozen=True, order=True)
class Constant:
    """Symbolic constant. value is a canonical RDF term string or an integer ID."""
    value: Union[str, int]
    def __str__(self) -> str:
        return str(self.value)

Term = Union[Variable, Constant]

@dataclass(frozen=True)
class Atom:
    predicate: str
    terms: Tuple[Term, ...]
    @property
    def arity(self) -> int:
        return len(self.terms)
    def __str__(self) -> str:
        return f"{self.predicate}({','.join(map(str, self.terms))})"

@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Tuple[Atom, ...]
    rule_id: str = ""
    def __str__(self) -> str:
        return f"{self.head} :- {', '.join(map(str, self.body))} ."

@dataclass(frozen=True)
class ConjunctiveQuery:
    answer_vars: Tuple[Variable, ...]
    body: Tuple[Atom, ...]
    name: str = "q"

@dataclass(frozen=True)
class GroundFact:
    predicate: str
    values: Tuple[int, ...]

@dataclass(frozen=True)
class QueryItem:
    source_atom_index: int
    atoms: Tuple[Atom, ...]
    source: str = "ORIGINAL"  # ORIGINAL | COT
    cot_rule_id: str | None = None
    # Query-frontier variables exposed by this source-atom alternative. A
    # frontier variable is an original answer variable or a variable shared
    # with another source CQ atom. All other COT-body variables are existential.
    interface_vars: Tuple[str, ...] = ()
    # Query variables unified with constants in the rule head.
    constant_bindings: Tuple[Tuple[str, Constant], ...] = ()
    # Complete MGU output mapping, including variable identifications.
    output_bindings: Tuple[Tuple[str, Term], ...] = ()

def variables_in_atoms(atoms: Iterable[Atom]) -> Tuple[Variable, ...]:
    seen = set(); out=[]
    for atom in atoms:
        for term in atom.terms:
            if isinstance(term, Variable) and term.name not in seen:
                seen.add(term.name); out.append(term)
    return tuple(out)
