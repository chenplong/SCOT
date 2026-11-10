"""SCOT redundancy removal and complete one-hop rule construction."""
import os
import random
import time
from collections import defaultdict, deque
from .core import (Atom, Const, Facts, Rule, apply_atom, canon_rule,
                   enumerate_derivations, fresh_rule, is_var,
                   remove_redundant, subsumes, unify_atoms, ground_row_key)
from .backend import (StageOneDelete, DfClosure, df_fixpoint_with_derive,
                      closure, one_hop, arities_of)

def _rule_violations(checker, R):
    try:
        sql = checker.violations_sql(R)
        out = checker.ctx.sql(sql).collect()
        return any((b.num_rows for b in out))
    except Exception:
        return None

def _scc_layers(omega):
    """SCCs of the predicate dependency graph, bottom-up (body layers first).

    Edge p -> q when a rule with head p has q in its body.  Returns
    (layers, comp_id) where layers[0] is the lowest layer (base predicates)
    and comp_id[p] is the index of p's SCC in layers.
    """
    preds = set()
    edges = defaultdict(set)
    for R in omega:
        preds.add(R.head.pred)
        for a in R.body:
            preds.add(a.pred)
            edges[R.head.pred].add(a.pred)
    seen, order = (set(), [])
    for s in preds:
        if s in seen:
            continue
        stack = [(s, iter(edges[s]))]
        seen.add(s)
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(edges[w])))
                    advanced = True
                    break
            if not advanced:
                order.append(v)
                stack.pop()
    redges = defaultdict(set)
    for p, qs in edges.items():
        for q in qs:
            redges[q].add(p)
    comp_id, comps = ({}, [])
    for s in reversed(order):
        if s in comp_id:
            continue
        cid = len(comps)
        cur, stack = ([], [s])
        comp_id[s] = cid
        while stack:
            v = stack.pop()
            cur.append(v)
            for w in redges[v]:
                if w not in comp_id:
                    comp_id[w] = cid
                    stack.append(w)
        comps.append(cur)
    cedges = defaultdict(set)
    for p, qs in edges.items():
        for q in qs:
            if comp_id[p] != comp_id[q]:
                cedges[comp_id[p]].add(comp_id[q])
    indeg = defaultdict(int)
    for a, bs in cedges.items():
        for b in bs:
            indeg[b] += 1
    ready = [c for c in range(len(comps)) if indeg[c] == 0]
    topo = []
    while ready:
        c = ready.pop()
        topo.append(c)
        for b in cedges[c]:
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
    bottom_up = [comps[c] for c in reversed(topo)]
    remap = {p: i for i, cs in enumerate(bottom_up) for p in cs}
    return (bottom_up, remap)

class COTCompressor:
    """Compress finite facts under positive safe Datalog rules.

    Facts use integer IDs or core.Const values. The input may be unclosed.
    G_c remains a subset of the supplied facts.
    """
    def __init__(self, facts, rules, T=2, Nb=4, seed=42, verbose=False, mem_gb=None):
        self.G_input = facts.copy() if isinstance(facts, Facts) else Facts(facts)
        self.rules = list(rules)
        self.T, self.K = int(T), int(Nb)
        if self.T < 1 or self.K < 1:
            raise ValueError("T and Nb must be at least 1")
        self.seed, self.verbose, self.mem_gb = seed, verbose, mem_gb
        self.stats = {}
        self.G_c = self.G_next = None
        self.W = {}
        self._validate()

    def _validate(self):
        arities_of(self.G_input, self.rules)
        for predicate, rows in self.G_input.rows.items():
            for row in rows:
                if not row or any(isinstance(term, bool) or
                                  not isinstance(term, (int, Const)) for term in row):
                    raise ValueError("Facts require positive-arity ground integer/Const tuples")
        for rule in self.rules:
            if not rule.body:
                raise ValueError("Rules must have nonempty bodies")
            variables = {term for atom in rule.body for term in atom.args if is_var(term)}
            if any(is_var(term) and term not in variables for term in rule.head.args):
                raise ValueError("Unbound head variable: %s" % rule)
            for atom in (rule.head,) + tuple(rule.body):
                if not atom.args or any(isinstance(term, bool) or
                                        not isinstance(term, (str, int, Const)) for term in atom.args):
                    raise ValueError("Invalid rule atom: %s" % atom)

    def log(self, *args, **kwargs):
        if self.verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)

    def _arities(self):
        return arities_of(self.G_star, self.rules)

    def _head_arities(self):
        return self._arities()

    def block_minimize(self, start):
        started = time.time()
        rng = random.Random(self.seed)
        current = start.copy()
        for round_index in range(self.T):
            facts = sorted(current.facts(), key=lambda f: (f[0], ground_row_key(f[1])))
            rng.shuffle(facts)
            block_size = max(1, (len(facts) + self.K - 1) // self.K)
            retained = Facts()
            recovered = DfClosure(self.rules, self._arities(), mem_gb=self.mem_gb)
            for offset in range(0, len(facts), block_size):
                new = [(pred, row) for pred, row in facts[offset:offset + block_size]
                       if not recovered.has(pred, row)]
                for pred, row in new:
                    retained.add(pred, row)
                if new:
                    recovered.add(new)
            current = retained
            self.log("[compression] block round %d/%d retained=%d" %
                     (round_index + 1, self.T, current.size()))
        self.stats["block_sec"] = time.time() - started
        return current

    def part1_input(self):
        started = time.time()
        deleter = StageOneDelete(self.rules, self.G_input.rows,
                                 mem_gb=self.mem_gb, log=self.log)
        eligible = [rule for rule in self.rules
                    if not any(atom.pred == rule.head.pred for atom in rule.body)]
        support = {rule: deleter._support(rule) for rule in eligible}
        layers, _ = _scc_layers(self.rules)
        for layer in reversed(layers):
            ranked = sorted((rule for rule in eligible if rule.head.pred in layer),
                            key=lambda rule: (-support[rule], str(canon_rule(rule))))
            for rule in ranked:
                pred = rule.head.pred
                before = len(deleter.rows.get(pred, ()))
                heads = deleter._facts(rule)
                deleter.rows.setdefault(pred, set()).difference_update(heads)
                removed = before - len(deleter.rows[pred])
                if removed:
                    deleter._register(pred)
                deleter.stats["deleted_1a"] += removed
        self.G_1a = Facts(deleter.rows)
        self.stats["deleted_1a"] = deleter.stats["deleted_1a"]
        self.stats["part1a_sec"] = time.time() - started
        self.stats["G_1a"] = self.G_1a.size()
        self.G_T = self.block_minimize(self.G_1a)
        return self.G_T

    def build_first_derive(self):
        started = time.time()
        rows, self.level, self.M = df_fixpoint_with_derive(
            self.rules, self.G_T.rows, self._arities(), mem_gb=self.mem_gb)
        derived = Facts(rows)
        self.sigma_star_ok = derived.size() == self.G_star.size() and all(
            derived.has(pred, row) for pred, row in self.G_star.facts())
        self.stats["fixpoint_ok"] = self.sigma_star_ok
        self.stats["levels"] = max(self.level.values(), default=0)
        self.stats["part21_sec"] = time.time() - started
        return self.M


    def _reasoning_rule(self, gamma):
        base = self.G_T
        queue = deque([(gamma, None, None)])
        trace = []
        while queue:
            f, parent_idx, pos = queue.popleft()
            R, gbody = self.M[f]
            for k, bf in enumerate(gbody):
                if not base.has(*bf):
                    queue.append((bf, len(trace), k))
            trace.append((R, gbody, parent_idx, pos))
        insts = []
        for R, gbody, parent_idx, pos in trace:
            Ri = fresh_rule(R)
            insts.append({'rule': Ri, 'sym_body': list(Ri.body), 'gbody': gbody, 'parent': parent_idx, 'pos': pos})
        gsub = {}

        def _resolve(t):
            seen = set()
            while is_var(t) and t in gsub and (t not in seen):
                seen.add(t)
                t = gsub[t]
            return t

        def _apply(a):
            return Atom(a.pred, tuple((_resolve(t) if is_var(t) else t for t in a.args)))
        for i in range(1, len(trace)):
            inst = insts[i]
            parent = insts[inst['parent']]
            target = _apply(parent['sym_body'][inst['pos']])
            sigma = unify_atoms(_apply(inst['rule'].head), target)
            if sigma is None:
                raise RuntimeError('reasoning-path unify failed')
            for k, v in sigma.items():
                gsub[k] = _resolve(v)
            inst['sym_body'] = [apply_atom(a, sigma) for a in inst['rule'].body]
        if gsub:
            for inst in insts:
                inst['sym_body'] = [_apply(a) for a in inst['sym_body']]
        body, gbody = ([], [])
        for inst in insts:
            for k, a in enumerate(inst['sym_body']):
                if base.has(*inst['gbody'][k]):
                    body.append(a)
                    gbody.append(inst['gbody'][k])
        head = insts[0]['rule'].head
        if gsub:
            head = _apply(head)
        bvars = set()
        for a in body:
            bvars.update(a.args)
        row = gamma[1]
        args = list(head.args)
        for i, t in enumerate(args):
            if is_var(t) and t not in bvars:
                args[i] = row[i]
        head = Atom(head.pred, tuple(args))
        if not all((not is_var(t) or t in bvars for t in head.args)):
            return None
        return (Rule(head, tuple(body)), tuple(gbody))

    def reasoning_paths(self):
        t0 = time.time()
        residual = sorted((f for f in self.G_star.facts() if not self.G_T.has(*f)),
                          key=lambda f: (f[0], ground_row_key(f[1])))
        uncovered = set(residual)
        pending = list(residual)
        path_rules, W = (set(), {})
        fact_rule = {}
        protected = []
        path_calls = 0
        while pending:
            f = pending.pop()
            if f not in uncovered:
                continue
            out = self._reasoning_rule(f)
            if out is None:
                protected.append(f)
                uncovered.discard(f)
                continue
            R, gbody = out
            key = canon_rule(R)
            path_rules.add(key)
            path_calls += 1
            covered = set()
            for head, body in enumerate_derivations(R, self.G_T):
                if head in uncovered:
                    covered.add(head)
                    W[head] = body
                    fact_rule[head] = key
            if f not in covered:
                covered.add(f)
                W[f] = gbody
                fact_rule[f] = key
            uncovered.difference_update(covered)
        unsound = self._gate_unsound(path_rules)
        if unsound:
            for f, key in list(fact_rule.items()):
                if key in unsound:
                    W.pop(f, None)
                    fact_rule.pop(f, None)
                    protected.append(f)
                    self.G_T.add(f[0], f[1])
            path_rules -= unsound
            self.stats['remaining_uncovered_after_gate'] = len(uncovered)
        for pred, row in protected:
            self.G_T.add(pred, row)
        self.covered = set(residual) - set(uncovered)
        self.residual = residual
        self.PathRules = path_rules
        self._fact_rule = fact_rule
        self.W = W
        self.stats['covered_by_Psi'] = 0
        self.stats['residual'] = len(residual)
        self.stats['protected_unsafe'] = len(protected)
        self.stats['unsound_rules'] = len(unsound)
        self.stats['PathRules'] = len(path_rules)
        self.stats['path_calls'] = path_calls
        self.stats['part24_sec'] = time.time() - t0
        self.log('[COT] residual=%d path_calls=%d PathRules=%d protected=%d unsound_rules=%d (%.2fs)' % (len(residual), path_calls, len(path_rules), len(protected), len(unsound), self.stats['part24_sec']), flush=True)
        return path_rules

    def _gate_unsound(self, rules):
        sigma = self._sigma_canon_set()
        budgets = [float(self.mem_gb if self.mem_gb is not None else os.environ.get('DF_MEM_GB', '8'))]
        last = None
        for mem in budgets:
            try:
                gate = StageOneDelete(list(rules), {p: set(v) for p, v in self.G_star.rows.items()}, mem_gb=mem)
                unsound = set()
                for R in rules:
                    if len(R.body) == 1 and R.body[0] == R.head:
                        continue
                    # Check the generated clause and its variable bindings.
                    if self._sound_by_rule_lib(R, sigma):
                        continue
                    viol = _rule_violations(gate, R)
                    if viol is None:
                        raise RuntimeError('gate evaluation failed for %s' % canon_rule(R))
                    if viol:
                        unsound.add(canon_rule(R))
                return unsound
            except Exception as exc:
                last = exc
        raise RuntimeError('soundness gate could not be evaluated: %s' % last)

    def _sigma_canon_set(self):
        s = getattr(self, '_sigma_canon', None)
        if s is None:
            s = {canon_rule(R) for R in self.rules}
            self._sigma_canon = s
        return s

    def _inv_pred_map(self):
        mapping = getattr(self, "_inv_map", None)
        if mapping is None:
            pairs = set()
            for rule in self.rules:
                if len(rule.body) != 1 or len(rule.head.args) != 2:
                    continue
                head, body = rule.head, rule.body[0]
                left, right = head.args
                if (head.pred == body.pred or len(body.args) != 2
                        or not is_var(left) or not is_var(right) or left == right):
                    continue
                if body.args == (right, left):
                    pairs.add((head.pred, body.pred))
            # Replacing a body predicate requires equivalence, so a single
            # directed inverse rule is insufficient for this shortcut.
            mapping = {head: body for head, body in sorted(pairs)
                       if (body, head) in pairs}
            self._inv_map = mapping
        return mapping

    def _transitive_preds(self):
        predicates = getattr(self, "_trans_preds", None)
        if predicates is None:
            predicates = set()
            for rule in self.rules:
                if len(rule.body) != 2 or len(rule.head.args) != 2:
                    continue
                first, second = rule.body
                if (first.pred != rule.head.pred or second.pred != rule.head.pred
                        or len(first.args) != 2 or len(second.args) != 2):
                    continue
                left, right = rule.head.args
                middle = first.args[1]
                variables = (left, middle, right)
                if not all(is_var(term) for term in variables) or len(set(variables)) != 3:
                    continue
                if first.args == (left, middle) and second.args == (middle, right):
                    predicates.add(rule.head.pred)
            self._trans_preds = predicates
        return predicates

    @staticmethod
    def _is_chain(atoms, head):
        if len(head.args) != 2 or not atoms:
            return False
        start, end = head.args
        outgoing, incoming = defaultdict(list), defaultdict(int)
        for index, atom in enumerate(atoms):
            if len(atom.args) != 2:
                return False
            left, right = atom.args
            outgoing[left].append((index, right))
            incoming[right] += 1
        for vertex in set(outgoing) | set(incoming):
            expected = (1 if vertex == start else 0) - (1 if vertex == end else 0)
            if len(outgoing[vertex]) - incoming[vertex] != expected:
                return False
        stack, trail = [start], []
        while stack:
            vertex = stack[-1]
            if outgoing[vertex]:
                _index, target = outgoing[vertex].pop()
                stack.append(target)
            else:
                trail.append(stack.pop())
        return len(trail) == len(atoms) + 1 and trail[0] == end and trail[-1] == start

    def _sound_by_rule_lib(self, R, sigma):
        if canon_rule(R) in sigma:
            return True
        inv = self._inv_pred_map()
        tpreds = self._transitive_preds()
        if len(R.body) >= 2 and all((len(a.args) == 2 for a in R.body)):
            preds = {a.pred for a in R.body}
            if len(preds) == 1:
                bp = next(iter(preds))
                if bp in tpreds:
                    if R.head.pred == bp and self._is_chain(list(R.body), R.head):
                        return True
                    if inv.get(R.head.pred) == bp and self._is_chain(list(R.body), Atom(bp, (R.head.args[1], R.head.args[0]))):
                        return True
        choices = []
        for a in R.body:
            p = inv.get(a.pred)
            if p is not None and len(a.args) == 2:
                choices.append([a, Atom(p, (a.args[1], a.args[0]))])
            else:
                choices.append([a])
        variants = 1
        for c in choices:
            variants *= len(c)
        if variants > 64:
            variants = 0
        tpreds = self._transitive_preds() if inv or variants else set()
        for mask in range(variants):
            combo, m = ([], mask)
            for c in choices:
                combo.append(c[m % len(c)])
                m //= len(c)
            if canon_rule(Rule(R.head, tuple(combo))) in sigma:
                return True
            if tpreds:
                preds = {a.pred for a in combo}
                if len(preds) == 1 and R.head.pred in preds and all((len(a.args) == 2 for a in combo)) and (next(iter(preds)) in tpreds) and self._is_chain(combo, R.head):
                    return True
        return False

    def build_omega(self):
        t0 = time.time()
        arities = self._head_arities()
        rules = []
        for p, n in sorted(arities.items()):
            vs = tuple(('X%d' % i for i in range(n)))
            a = Atom(p, vs)
            rules.append(Rule(a, (a,)))
        rules.extend(sorted(self.PathRules, key=lambda rule: (
            rule.head.pred, ground_row_key(rule.head.args),
            tuple((atom.pred, ground_row_key(atom.args)) for atom in rule.body))))
        self.omega = remove_redundant(rules)
        unsound = self._gate_unsound(self.omega)
        if unsound:
            kept = [R for R in self.omega if canon_rule(R) not in unsound]
            if hasattr(self, '_fact_rule'):
                for f, key in self._fact_rule.items():
                    if key in unsound:
                        self.G_T.add(f[0], f[1])
            self.omega = kept
            self.log('[COT] omega gate: dropped %d unsound rules' % len(unsound), flush=True)
            self.stats['omega_unsound'] = len(unsound)
        fact_rule = getattr(self, '_fact_rule', {})
        omega_rules = list(self.omega)
        omega_keys = {canon_rule(R) for R in omega_rules}
        sub_cache = {}

        def _covered_key(key):
            if key in omega_keys:
                return True
            if key in sub_cache:
                return sub_cache[key]
            ok = any((subsumes(R, key) for R in omega_rules))
            sub_cache[key] = ok
            return ok
        kept_back = 0
        for f in getattr(self, 'residual', []):
            if self.G_T.has(*f):
                continue
            key = fact_rule.get(f)
            if key is None or not _covered_key(key):
                self.G_T.add(f[0], f[1])
                kept_back += 1
        self.stats['kept_for_coverage'] = kept_back
        self.log('[COT] coverage (witness-based over %d Omega rules): kept back %d facts' % (len(self.omega), kept_back), flush=True)
        self.omega_by_pred = defaultdict(set)
        for rule in self.omega:
            self.omega_by_pred[rule.head.pred].add(rule)
        self.stats["omega"] = len(self.omega)
        self.stats["part25_sec"] = time.time() - t0
        return self.omega

    def check_coverage(self):
        if self.G_c is None:
            raise RuntimeError("Run compression before checking coverage")
        restored = one_hop(self.G_c, self.omega, mem_gb=self.mem_gb)
        return restored.size() == self.G_star.size() and all(
            restored.has(pred, row) for pred, row in self.G_star.facts())

    def run(self):
        started = time.time()
        self.G_star = closure(self.G_input, self.rules, mem_gb=self.mem_gb)
        self.part1_input()
        self.build_first_derive()
        if not self.sigma_star_ok:
            self.G_T = self.G_1a.copy()
            self.build_first_derive()
        if not self.sigma_star_ok:
            self.G_T = self.G_input.copy()
            self.build_first_derive()
        if not self.sigma_star_ok:
            raise RuntimeError("Retained graph does not preserve the original closure")
        self.reasoning_paths()
        self.build_omega()
        if not all(self.G_input.has(pred, row) for pred, row in self.G_T.facts()):
            raise RuntimeError("Compression retained a fact outside the input graph")
        self.G_c = self.G_next = self.G_T.copy()
        self.Omega = self.omega
        self.coverage_ok = self.check_coverage()
        if not self.coverage_ok:
            raise RuntimeError("COT rules failed exact one-hop recovery")
        retained = self.G_c.size() / self.G_star.size() if self.G_star.size() else 1.0
        self.stats.update(G_input=self.G_input.size(), G_star=self.G_star.size(),
                          G_c=self.G_c.size(), G_next=self.G_c.size(), omega=len(self.omega),
                          compression_ratio=retained, removed_fraction=1.0 - retained,
                          coverage_ok=True, build_sec=time.time() - started)
        return self
