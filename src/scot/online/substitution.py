from __future__ import annotations
from typing import Dict, Optional
from .model import Atom, Constant, Rule, Term, Variable

Substitution=Dict[str,Term]

def deref(t:Term,s:Substitution)->Term:
    seen=set()
    while isinstance(t,Variable) and t.name in s and t.name not in seen:
        seen.add(t.name); t=s[t.name]
    return t

def unify_terms(a:Term,b:Term,s:Substitution)->bool:
    a=deref(a,s); b=deref(b,s)
    if a==b: return True
    if isinstance(a,Variable): s[a.name]=b; return True
    if isinstance(b,Variable): s[b.name]=a; return True
    return False

def unify_atoms(a:Atom,b:Atom,initial:Optional[Substitution]=None)->Optional[Substitution]:
    if a.predicate!=b.predicate or a.arity!=b.arity: return None
    s=dict(initial or {})
    for x,y in zip(a.terms,b.terms):
        if not unify_terms(x,y,s): return None
    # Normalize chained variables.
    for k in list(s): s[k]=deref(s[k],s)
    return s



def unify_atoms_inplace(a:Atom,b:Atom,s:Substitution)->bool:
    """Unify two atoms into ``s`` without copying or normalizing it.

    This is intended for hot paths that accumulate one monotonic substitution
    across many guaranteed-compatible proof-shape edges.  Newly-created
    bindings are rolled back if a later term in the same atom conflicts, so a
    failed call leaves ``s`` unchanged.  Existing bindings are never overwritten.
    """
    if a.predicate!=b.predicate or a.arity!=b.arity:
        return False
    added=[]
    for x,y in zip(a.terms,b.terms):
        x=deref(x,s); y=deref(y,s)
        if x==y:
            continue
        if isinstance(x,Variable):
            s[x.name]=y; added.append(x.name); continue
        if isinstance(y,Variable):
            s[y.name]=x; added.append(y.name); continue
        for name in reversed(added):
            del s[name]
        return False
    return True

def apply_term(t:Term,s:Substitution)->Term:
    return deref(t,s) if isinstance(t,Variable) else t

def apply_atom(a:Atom,s:Substitution)->Atom:
    return Atom(a.predicate,tuple(apply_term(t,s) for t in a.terms))

def apply_rule(r:Rule,s:Substitution)->Rule:
    return Rule(apply_atom(r.head,s),tuple(apply_atom(a,s) for a in r.body),r.rule_id)

def standardize_rule(r:Rule,prefix:str,forbidden_names=())->Rule:
    names={}
    used=set(forbidden_names)
    def f(t):
        if not isinstance(t,Variable): return t
        if t.name not in names:
            candidate=f'{prefix}_{t.name}'
            while candidate in used:
                candidate += '_fresh'
            used.add(candidate)
            names[t.name]=Variable(candidate)
        return names[t.name]
    def a(x): return Atom(x.predicate,tuple(f(t) for t in x.terms))
    return Rule(a(r.head),tuple(a(x) for x in r.body),r.rule_id)
