"""Factorized P00/P01/P10/P11 SQL compilation over retained facts."""
from __future__ import annotations
from typing import Callable, Sequence, Union
from .model import Atom, Constant, QueryItem, Variable

# A representative variable can inherit several original frontier filters.
# Keep each view's original column name when MGU output columns are identified.
UnaryFilters = dict[str, Union[str, tuple[tuple[str, str], ...]]]


class EmptyQuery(Exception):
    pass


class CQSqlCompiler:
    """Compile atom-wise recovery alternatives, then join their frontiers."""

    def __init__(
        self,
        table_by_predicate: dict[str, str],
        constant_lookup: Callable[[str], int | None],
        *,
        query_frontier_sensitive_execution: bool = True,
        unary_frontier_absorption: bool = True,
    ):
        self.tables = table_by_predicate
        self.constant_lookup = constant_lookup
        self.query_frontier_sensitive_execution = bool(query_frontier_sensitive_execution)
        self.unary_frontier_absorption = bool(unary_frontier_absorption)

    @staticmethod
    def _unary_filter_columns(views: UnaryFilters | None,
                              var_aliases: dict[str, str]):
        return {
            name: ((value, var_aliases[name]),) if isinstance(value, str) else value
            for name, value in (views or {}).items()
        }


    def _resolve_atoms(self, atoms: Sequence[Atom]):
        resolved = []
        for original_index, atom in enumerate(atoms):
            table = self.tables.get(atom.predicate)
            if table is None:
                raise EmptyQuery(atom.predicate)
            const_count = 0
            terms = []
            for term in atom.terms:
                if isinstance(term, Constant):
                    const_count += 1
                    value = int(term.value) if isinstance(term.value, int) else self.constant_lookup(str(term.value))
                    if value is None:
                        raise EmptyQuery(str(term.value))
                    terms.append(('c', int(value)))
                else:
                    terms.append(('v', term.name))
            resolved.append(
                {
                    'index': original_index,
                    'atom': atom,
                    'table': table,
                    'terms': tuple(terms),
                    'vars': {v for kind, v in terms if kind == 'v'},
                    'const_count': const_count,
                }
            )
        return resolved


    @staticmethod
    def _answer_aware_order(resolved, initially_bound: set[str] | None = None):
        """Order one COT body deterministically using bound variables.

        Unary-frontier bindings seed the bound-variable set, prioritizing atoms
        that share those variables.
        """
        if not resolved:
            return []
        bound = set(initially_bound or ())
        remaining = list(resolved)
        order = []
        while remaining:
            connected = [r for r in remaining if r['vars'] & bound]
            if connected:
                nxt = max(
                    connected,
                    key=lambda r: (
                        len(r['vars'] & bound),
                        r['const_count'],
                        int(r['vars'] <= bound),
                        -len(r['vars'] - bound),
                        -r['index'],
                    ),
                )
            else:
                nxt = max(
                    remaining,
                    key=lambda r: (r['const_count'], -len(r['vars']), -r['index']),
                )
            order.append(nxt)
            remaining.remove(nxt)
            bound.update(nxt['vars'])
        return order


    def _compile_candidate_branch_frontier_sensitive(
        self,
        item: QueryItem,
        *,
        group_index: int,
        candidate_index: int,
        var_aliases: dict[str, str],
        absorbed_unary_views: UnaryFilters | None = None,
    ) -> tuple[list[str], str, dict[str, int]]:
        """Compile one C_{i,j} using its query-frontier variables as answers.

        At each stage only frontier variables and variables needed by future
        atoms survive.  Variables local to the COT body are eliminated at their
        last use; atoms that introduce no future/frontier value become EXISTS
        guards.  The returned SELECT has the same schema for every branch in an
        atom group, making UNION ALL safe and deterministic.
        """
        frontier = tuple(item.interface_vars)
        const_map = dict(item.constant_bindings)
        atoms = tuple(item.atoms)
        if not atoms:
            if frontier and not set(frontier) <= set(const_map):
                raise ValueError('empty COT body cannot bind query-frontier variables')
            return [], 'SELECT 1 AS __one', {
                'stages': 0,
                'inner_joins': 0,
                'semi_joins': 0,
                'cross_joins': 0,
                'eliminated_vars': 0,
                'early_projections': 0,
                'absorbed_unary_filters': 0,
            }

        absorbed_unary_views = self._unary_filter_columns(absorbed_unary_views, var_aliases)
        resolved = self._resolve_atoms(atoms)
        frontier_set = set(frontier) - set(const_map)
        # A unary view on a constant-bound frontier variable still has to be
        # checked: the constant is a value, not an exemption from V_x.
        absorbed_unary_views = {
            name: view for name, view in absorbed_unary_views.items()
            if name in set(frontier)
        }
        order = self._answer_aware_order(resolved, set(absorbed_unary_views))
        present = set().union(*(r['vars'] for r in order))
        missing = frontier_set - present
        if missing:
            raise ValueError(f'Unbound query-frontier variables in candidate: {sorted(missing)}')

        prefix = f'f_g{group_index}_c{candidate_index}_s'
        ctes: list[str] = []
        prior: tuple[str, ...] = ()
        joins = semis = crosses = eliminated = early_projection = 0
        absorption_filters = 0
        absorption_applied: set[str] = set()
        constant_absorption: list[str] = []
        for name in sorted(set(absorbed_unary_views) & set(const_map)):
            value = const_map[name].value
            expr = int(value) if isinstance(value, int) else self.constant_lookup(str(value))
            for view, column in absorbed_unary_views[name]:
                constant_absorption.append(
                    f'EXISTS (SELECT 1 FROM {view} uv WHERE uv.{column} = {expr})'
                )
                absorption_filters += 1
            absorption_applied.add(name)

        for position, r in enumerate(order):
            stage_name = f'{prefix}{position}'
            alias = f'r{position}'
            prior_set = set(prior)
            future = (
                set().union(*(x['vars'] for x in order[position + 1:]))
                if position + 1 < len(order)
                else set()
            )
            live = future | frontier_set
            newly_bound = []
            local = {}
            join_conditions = []
            local_conditions = []
            for j, (kind, value) in enumerate(r['terms']):
                col = f'{alias}.arg{j}'
                if kind == 'c':
                    local_conditions.append(f'{col} = {value}')
                elif value in prior_set:
                    join_conditions.append(f'{col} = p.c{prior.index(value)}')
                elif value in local:
                    local_conditions.append(f'{col} = {local[value]}')
                else:
                    local[value] = col
                    newly_bound.append(value)

            # Unary-frontier absorption: V_x is a set-valued one-column view.
            # Apply it exactly once, at the first chosen body atom carrying x,
            # as a correlated EXISTS (semi-join) so no duplicate multiplicity
            # or cross product is introduced.
            for name in sorted((r['vars'] & set(absorbed_unary_views)) - absorption_applied):
                if name in prior_set:
                    expr = f'p.c{prior.index(name)}'
                else:
                    expr = local[name]
                for view, column in absorbed_unary_views[name]:
                    local_conditions.append(
                        f'EXISTS (SELECT 1 FROM {view} uv WHERE uv.{column} = {expr})'
                    )
                    absorption_filters += 1
                absorption_applied.add(name)

            if position == 0 and constant_absorption:
                local_conditions.extend(constant_absorption)

            available = prior + tuple(newly_bound)
            keep = tuple(v for v in available if v in live)
            eliminated += len(available) - len(keep)
            early_projection += int(len(keep) < len(available))
            projected = (
                ', '.join(
                    f'{("p.c" + str(prior.index(v))) if v in prior_set else local[v]} AS c{i}'
                    for i, v in enumerate(keep)
                )
                if keep
                else '1 AS __one'
            )

            if position == 0:
                sql = f'SELECT DISTINCT {projected} FROM {r["table"]} {alias}'
                if local_conditions:
                    sql += ' WHERE ' + ' AND '.join(local_conditions)
            elif not any(v in live for v in newly_bound):
                semis += 1
                tests = join_conditions + local_conditions
                guard = f'SELECT 1 FROM {r["table"]} {alias}'
                if tests:
                    guard += ' WHERE ' + ' AND '.join(tests)
                sql = f'SELECT DISTINCT {projected} FROM {prefix}{position - 1} p WHERE EXISTS ({guard})'
            else:
                if join_conditions:
                    joins += 1
                    source = (
                        f'{prefix}{position - 1} p INNER JOIN {r["table"]} {alias} ON '
                        + ' AND '.join(join_conditions)
                    )
                else:
                    crosses += 1
                    source = f'{prefix}{position - 1} p CROSS JOIN {r["table"]} {alias}'
                sql = f'SELECT DISTINCT {projected} FROM {source}'
                if local_conditions:
                    sql += ' WHERE ' + ' AND '.join(local_conditions)
            ctes.append(f'{stage_name} AS ({sql})')
            prior = keep

        last = f'{prefix}{len(order) - 1}'
        if frontier:
            cols = []
            for name in frontier:
                if name in const_map:
                    value = const_map[name].value
                    lit = int(value) if isinstance(value, int) else self.constant_lookup(str(value)); cols.append(f'CAST({lit} AS BIGINT) AS {var_aliases[name]}')
                elif name not in prior:
                    raise ValueError(f'Frontier variable {name} was eliminated too early')
                else:
                    cols.append(f'CAST(c{prior.index(name)} AS BIGINT) AS {var_aliases[name]}')
            branch_sql = 'SELECT DISTINCT ' + ', '.join(cols) + f' FROM {last}'
        else:
            branch_sql = f'SELECT 1 AS __one FROM {last} LIMIT 1'

        return ctes, branch_sql, {
            'stages': len(order),
            'inner_joins': joins,
            'semi_joins': semis,
            'cross_joins': crosses,
            'eliminated_vars': eliminated,
            'early_projections': early_projection,
            'absorbed_unary_filters': absorption_filters,
        }


    def _compile_candidate_branch_flat(
        self,
        item: QueryItem,
        *,
        group_index: int,
        candidate_index: int,
        var_aliases: dict[str, str],
        absorbed_unary_views: UnaryFilters | None = None,
    ) -> tuple[list[str], str, dict[str, int]]:
        """Compile one candidate without query-frontier-sensitive staging.

        The body is kept as one ordinary CQ relation and only the final SELECT
        projects to the query-frontier schema.  No frontier-guided atom order,
        early existential-variable elimination, intermediate projection, or
        staged semi-join conversion is performed.  This is the ablation
        baseline for query-frontier-sensitive execution.

        Unary-frontier absorption, when enabled independently, remains a
        logical correlated EXISTS restriction on the complete candidate body.
        """
        frontier = tuple(item.interface_vars)
        const_map = dict(item.constant_bindings)
        atoms = tuple(item.atoms)
        if not atoms:
            if frontier and not set(frontier) <= set(const_map):
                raise ValueError('empty COT body cannot bind query-frontier variables')
            return [], 'SELECT 1 AS __one', {
                'stages': 0,
                'inner_joins': 0,
                'semi_joins': 0,
                'cross_joins': 0,
                'eliminated_vars': 0,
                'early_projections': 0,
                'absorbed_unary_filters': 0,
            }

        absorbed_unary_views = self._unary_filter_columns(absorbed_unary_views, var_aliases)
        frontier_set = set(frontier) - set(const_map)
        absorbed_unary_views = {
            name: view for name, view in absorbed_unary_views.items()
            if name in set(frontier)
        }

        aliases: list[tuple[str, str]] = []
        bindings: dict[str, str] = {}
        conditions: list[str] = []
        present: set[str] = set()
        connected_joins = 0
        cross_joins = 0

        for ai, atom in enumerate(atoms):
            prior_names = set(bindings)
            table = self.tables.get(atom.predicate)
            if table is None:
                raise EmptyQuery(atom.predicate)
            alias = f'r{ai}'
            aliases.append((table, alias))
            atom_shared = False
            local: dict[str, str] = {}
            for j, term in enumerate(atom.terms):
                col = f'{alias}.arg{j}'
                if isinstance(term, Constant):
                    value = int(term.value) if isinstance(term.value, int) else self.constant_lookup(str(term.value))
                    if value is None:
                        raise EmptyQuery(str(term.value))
                    conditions.append(f'CAST({col} AS BIGINT) = {int(value)}')
                    continue
                name = term.name
                present.add(name)
                if name in prior_names:
                    conditions.append(f'{col} = {bindings[name]}')
                    atom_shared = True
                elif name in local:
                    conditions.append(f'{col} = {local[name]}')
                else:
                    local[name] = col
                    bindings[name] = col
            if ai:
                if atom_shared:
                    connected_joins += 1
                else:
                    cross_joins += 1

        present |= set(const_map)
        missing = frontier_set - present
        if missing:
            raise ValueError(f'Unbound query-frontier variables in candidate: {sorted(missing)}')

        absorption_filters = 0
        for name, views in sorted(absorbed_unary_views.items()):
            if name in const_map:
                value = const_map[name].value
                expr = int(value) if isinstance(value, int) else self.constant_lookup(str(value))
            else:
                expr = bindings[name]
            for view, column in views:
                conditions.append(
                    f'EXISTS (SELECT 1 FROM {view} uv WHERE uv.{column} = {expr})'
                )
                absorption_filters += 1

        from_sql = ' CROSS JOIN '.join(f'{table} {alias}' for table, alias in aliases)
        if frontier:
            parts = []
            for name in frontier:
                if name in const_map:
                    value = const_map[name].value
                    lit = int(value) if isinstance(value, int) else self.constant_lookup(str(value))
                    parts.append(f'CAST({lit} AS BIGINT) AS {var_aliases[name]}')
                else:
                    parts.append(f'CAST({bindings[name]} AS BIGINT) AS {var_aliases[name]}')
            branch_sql = f'SELECT DISTINCT {", ".join(parts)} FROM {from_sql}'
        else:
            branch_sql = f'SELECT 1 AS __one FROM {from_sql}'
        if conditions:
            branch_sql += ' WHERE ' + ' AND '.join(conditions)
        if not frontier:
            branch_sql += ' LIMIT 1'

        return [], branch_sql, {
            'stages': 1,
            'inner_joins': connected_joins,
            'semi_joins': 0,
            'cross_joins': cross_joins,
            'eliminated_vars': 0,
            'early_projections': 0,
            'absorbed_unary_filters': absorption_filters,
        }


    def _compile_candidate_branch(
        self,
        item: QueryItem,
        *,
        group_index: int,
        candidate_index: int,
        var_aliases: dict[str, str],
        absorbed_unary_views: UnaryFilters | None = None,
    ) -> tuple[list[str], str, dict[str, int]]:
        # The MGU can identify two different query columns. Compile on its
        # representative variables, then restore every original output name.
        mapping = dict(item.output_bindings)
        if any(isinstance(term, Variable) and term.name != name
               for name, term in mapping.items()):
            targets = {name: mapping.get(name, Variable(name))
                       for name in item.interface_vars}
            representatives = tuple(dict.fromkeys(
                term.name for term in targets.values()
                if isinstance(term, Variable)))
            constants = tuple((name, term) for name, term in targets.items()
                              if isinstance(term, Constant))
            normalized_frontier = representatives + tuple(name for name, _ in constants)
            aliases = {name: f'__mapped_{i}'
                       for i, name in enumerate(normalized_frontier)}
            # Push all original filters onto their MGU representative before
            # QFSE orders/joins/projects the body. Distinct source columns can
            # identify with one variable, so concatenation is essential here.
            mapped_views = {}
            for name, views in self._unary_filter_columns(
                    absorbed_unary_views, var_aliases).items():
                if name not in targets:
                    continue
                target = targets[name]
                representative = target.name if isinstance(target, Variable) else name
                mapped_views[representative] = mapped_views.get(representative, ()) + views
            normalized = QueryItem(item.source_atom_index, item.atoms,
                                   item.source, item.cot_rule_id,
                                   normalized_frontier, constants)
            ctes, body_sql, stats = self._compile_candidate_branch(
                normalized, group_index=group_index,
                candidate_index=candidate_index, var_aliases=aliases,
                absorbed_unary_views=mapped_views)
            expressions = {}
            for name, term in targets.items():
                if isinstance(term, Variable):
                    expressions[name] = f'mapped.{aliases[term.name]}'
                else:
                    literal = (int(term.value) if isinstance(term.value, int)
                               else self.constant_lookup(str(term.value)))
                    if literal is None:
                        raise EmptyQuery(str(term.value))
                    expressions[name] = str(literal)
            columns = ', '.join(f'{expr} AS {var_aliases[name]}'
                                for name, expr in expressions.items())
            branch_sql = f'SELECT DISTINCT {columns} FROM ({body_sql}) mapped'
            return ctes, branch_sql, stats
        if self.query_frontier_sensitive_execution:
            return self._compile_candidate_branch_frontier_sensitive(
                item,
                group_index=group_index,
                candidate_index=candidate_index,
                var_aliases=var_aliases,
                absorbed_unary_views=absorbed_unary_views,
            )
        return self._compile_candidate_branch_flat(
            item,
            group_index=group_index,
            candidate_index=candidate_index,
            var_aliases=var_aliases,
            absorbed_unary_views=absorbed_unary_views,
        )


    @staticmethod
    def _group_order(groups):
        if not groups:
            return []
        first = max(
            groups,
            key=lambda g: (g['constants'], bool(g['vars']), -len(g['vars']), -g['index']),
        )
        order = [first]
        remaining = [g for g in groups if g is not first]
        bound = set(first['vars'])
        while remaining:
            connected = [g for g in remaining if g['vars'] & bound]
            if connected:
                nxt = max(
                    connected,
                    key=lambda g: (
                        int(g['vars'] <= bound),
                        g['constants'],
                        len(g['vars'] & bound),
                        -len(g['vars'] - bound),
                        -g['index'],
                    ),
                )
            else:
                nxt = max(
                    remaining,
                    key=lambda g: (int(not g['vars']), g['constants'], -len(g['vars']), -g['index']),
                )
            order.append(nxt)
            remaining.remove(nxt)
            bound.update(nxt['vars'])
        return order


    def _compile_factorized_direct(
        self,
        candidate_groups: Sequence[Sequence[QueryItem]],
        projection: Sequence[Variable] = (),
    ) -> tuple[str, dict[str, int]]:
        """Compile one concrete factorized COT plan under the current switches.

        Query-frontier-sensitive execution controls candidate-internal staged
        projection/elimination. Unary-frontier absorption controls whether
        singleton-frontier factor UCQs are consolidated into V_x views and
        pushed into multi-frontier branches. Both transformations are exact
        under set semantics and neither enumerates complete COT combinations.
        """
        groups = tuple(tuple(group) for group in candidate_groups)
        empty_plan = {
            'atom_groups': 0,
            'candidate_branches': 0,
            'active_candidate_branches': 0,
            'static_empty_branches': 0,
            'frontier_variables_total': 0,
            'candidate_stages': 0,
            'candidate_inner_joins': 0,
            'candidate_semi_joins': 0,
            'candidate_cross_joins': 0,
            'candidate_eliminated_vars': 0,
            'candidate_early_projections': 0,
            'candidate_absorbed_unary_filters': 0,
            'unary_frontier_source_groups': 0,
            'unary_frontier_views': 0,
            'unary_frontier_consolidation_joins': 0,
            'absorbed_unary_views': 0,
            'unary_absorption_edges': 0,
            'residual_unary_views': 0,
            'residual_unary_guards': 0,
            'zero_frontier_groups': 0,
            'top_level_groups_after_absorption': 0,
            'global_inner_joins': 0,
            'global_semi_joins': 0,
            'global_cross_joins': 0,
        }
        if not groups:
            if projection:
                raise ValueError('non-empty answer projection for an empty CQ')
            return 'SELECT 1 AS one', empty_plan

        group_frontiers: list[tuple[str, ...]] = []
        var_order: list[str] = []
        seen_vars: set[str] = set()
        for gi, group in enumerate(groups):
            if not group:
                raise EmptyQuery(f'source atom {gi} has no Original/COT alternatives')
            frontier = tuple(group[0].interface_vars)
            if any(tuple(item.interface_vars) != frontier for item in group):
                raise ValueError('all alternatives of one source atom must expose the same frontier')
            group_frontiers.append(frontier)
            for name in frontier:
                if name not in seen_vars:
                    seen_vars.add(name)
                    var_order.append(name)
        var_aliases = {name: f'qv{i}' for i, name in enumerate(var_order)}

        all_ctes: list[str] = []
        total_candidates = active_candidates = static_empty = 0
        candidate_stages = candidate_joins = candidate_semis = candidate_crosses = 0
        eliminated = early_projection = absorbed_filters = 0

        def compile_group(
            gi: int,
            group: tuple[QueryItem, ...],
            absorbed_views: dict[str, str] | None = None,
        ) -> dict:
            nonlocal total_candidates, active_candidates, static_empty
            nonlocal candidate_stages, candidate_joins, candidate_semis, candidate_crosses
            nonlocal eliminated, early_projection, absorbed_filters

            frontier = group_frontiers[gi]
            branch_sqls: list[str] = []
            for cj, item in enumerate(group):
                total_candidates += 1
                try:
                    branch_ctes, branch_sql, meta = self._compile_candidate_branch(
                        item,
                        group_index=gi,
                        candidate_index=cj,
                        var_aliases=var_aliases,
                        absorbed_unary_views=absorbed_views,
                    )
                except EmptyQuery:
                    static_empty += 1
                    continue
                all_ctes.extend(branch_ctes)
                branch_sqls.append(branch_sql)
                active_candidates += 1
                candidate_stages += meta['stages']
                candidate_joins += meta['inner_joins']
                candidate_semis += meta['semi_joins']
                candidate_crosses += meta['cross_joins']
                eliminated += meta['eliminated_vars']
                early_projection += meta['early_projections']
                absorbed_filters += meta.get('absorbed_unary_filters', 0)

            if not branch_sqls:
                raise EmptyQuery(f'all alternatives for source atom {gi} are statically empty')

            union_name = f'f_g{gi}_union'
            group_name = f'f_g{gi}'
            if not frontier:
                branch_sqls = ['(%s)' % sql for sql in branch_sqls]
            all_ctes.append(f'{union_name} AS (' + ' UNION ALL '.join(branch_sqls) + ')')
            if frontier:
                cols = ', '.join(var_aliases[name] for name in frontier)
                all_ctes.append(f'{group_name} AS (SELECT DISTINCT {cols} FROM {union_name})')
            else:
                all_ctes.append(f'{group_name} AS (SELECT 1 AS __one FROM {union_name} LIMIT 1)')

            original = group[0].atoms[0] if group[0].source == 'ORIGINAL' and len(group[0].atoms) == 1 else None
            constants = 0 if original is None else sum(isinstance(term, Constant) for term in original.terms)
            return {
                'index': gi,
                'name': group_name,
                'frontier': frontier,
                'vars': set(frontier),
                'constants': constants,
            }

        # Ablation path: factorized evaluation without unary-frontier
        # consolidation/absorption. Every source-atom UCQ remains an independent
        # top-level factor, matching the pre-absorption factorized architecture.
        if not self.unary_frontier_absorption:
            relation_groups: list[dict] = []
            guards: list[str] = []
            zero_groups = 0
            for gi, group in enumerate(groups):
                meta = compile_group(gi, group)
                if meta['frontier']:
                    relation_groups.append(meta)
                else:
                    guards.append(f'EXISTS (SELECT 1 FROM {meta["name"]})')
                    zero_groups += 1

            global_joins = global_semis = global_crosses = 0
            bindings: dict[str, str] = {}
            from_sql = ''
            if relation_groups:
                ordered = self._group_order(relation_groups)
                first = ordered[0]
                first_alias = 'tg0'
                from_sql = f'{first["name"]} {first_alias}'
                bindings = {
                    name: f'{first_alias}.{var_aliases[name]}'
                    for name in first['frontier']
                }

                for pos, group in enumerate(ordered[1:], 1):
                    alias = f'tg{pos}'
                    shared = [name for name in group['frontier'] if name in bindings]
                    new = [name for name in group['frontier'] if name not in bindings]
                    if shared and not new:
                        cond = ' AND '.join(
                            f'{alias}.{var_aliases[name]} = {bindings[name]}' for name in shared
                        )
                        guards.append(
                            f'EXISTS (SELECT 1 FROM {group["name"]} {alias} WHERE {cond})'
                        )
                        global_semis += 1
                        continue
                    if shared:
                        cond = ' AND '.join(
                            f'{alias}.{var_aliases[name]} = {bindings[name]}' for name in shared
                        )
                        from_sql += f' INNER JOIN {group["name"]} {alias} ON {cond}'
                        global_joins += 1
                    else:
                        from_sql += f' CROSS JOIN {group["name"]} {alias}'
                        global_crosses += 1
                    for name in new:
                        bindings[name] = f'{alias}.{var_aliases[name]}'

            if projection:
                cols = []
                for i, var in enumerate(projection):
                    expr = bindings.get(var.name)
                    if expr is None:
                        raise ValueError(f'Unbound answer variable {var.name}')
                    cols.append(f'{expr} AS a{i}')
                select = 'SELECT DISTINCT ' + ', '.join(cols)
            else:
                select = 'SELECT DISTINCT 1 AS one'

            with_sql = 'WITH ' + ', '.join(all_ctes) + ' '
            sql = with_sql + (f'{select} FROM {from_sql}' if from_sql else select)
            if guards:
                sql += ' WHERE ' + ' AND '.join(guards)

            return sql, {
                'atom_groups': len(groups),
                'candidate_branches': total_candidates,
                'active_candidate_branches': active_candidates,
                'static_empty_branches': static_empty,
                'frontier_variables_total': sum(len(f) for f in group_frontiers),
                'candidate_stages': candidate_stages,
                'candidate_inner_joins': candidate_joins,
                'candidate_semi_joins': candidate_semis,
                'candidate_cross_joins': candidate_crosses,
                'candidate_eliminated_vars': eliminated,
                'candidate_early_projections': early_projection,
                'candidate_absorbed_unary_filters': 0,
                'unary_frontier_source_groups': 0,
                'unary_frontier_views': 0,
                'unary_frontier_consolidation_joins': 0,
                'absorbed_unary_views': 0,
                'unary_absorption_edges': 0,
                'residual_unary_views': 0,
                'residual_unary_guards': 0,
                'zero_frontier_groups': zero_groups,
                'top_level_groups_after_absorption': len(relation_groups),
                'global_inner_joins': global_joins,
                'global_semi_joins': global_semis,
                'global_cross_joins': global_crosses,
            }

        # Phase 1a: source factors with zero/singleton frontier are independent
        # of multi-frontier absorption, so compile them first.
        unary_groups_by_var: dict[str, list[dict]] = {}
        zero_groups: list[dict] = []
        compiled_meta: dict[int, dict] = {}
        for gi, group in enumerate(groups):
            frontier = group_frontiers[gi]
            if len(frontier) > 1:
                continue
            meta = compile_group(gi, group)
            compiled_meta[gi] = meta
            if not frontier:
                zero_groups.append(meta)
            else:
                unary_groups_by_var.setdefault(frontier[0], []).append(meta)

        # Phase 1b: consolidate all singleton-frontier UCQs on the same variable
        # into one set-valued unary view V_x.  No branch combinations are formed.
        unary_view_by_var: dict[str, str] = {}
        unary_consolidation_joins = 0
        for name in var_order:
            sources = unary_groups_by_var.get(name, ())
            if not sources:
                continue
            col = var_aliases[name]
            view_name = f'f_uv_{col}'
            first = sources[0]
            from_sql = f'{first["name"]} u0'
            for pos, src in enumerate(sources[1:], 1):
                from_sql += (
                    f' INNER JOIN {src["name"]} u{pos}'
                    f' ON u{pos}.{col} = u0.{col}'
                )
                unary_consolidation_joins += 1
            all_ctes.append(
                f'{view_name} AS (SELECT DISTINCT u0.{col} AS {col} FROM {from_sql})'
            )
            unary_view_by_var[name] = view_name

        # Phase 2: compile every multi-frontier factor with all available unary
        # domain views absorbed into every branch carrying the corresponding
        # variable.  This is the distributive rewrite V_x ⋈ (⋃C_j)=⋃(V_x⋈C_j).
        multi_metas: list[dict] = []
        multi_vars: set[str] = set()
        unary_absorption_edges = 0
        absorbed_view_names: set[str] = set()
        for gi, group in enumerate(groups):
            frontier = group_frontiers[gi]
            if len(frontier) <= 1:
                continue
            absorbed = {
                name: unary_view_by_var[name]
                for name in frontier
                if name in unary_view_by_var
            }
            unary_absorption_edges += len(absorbed)
            absorbed_view_names.update(absorbed.values())
            meta = compile_group(gi, group, absorbed)
            compiled_meta[gi] = meta
            multi_metas.append(meta)
            multi_vars.update(frontier)

        answer_names = {v.name for v in projection}
        guards: list[str] = [f'EXISTS (SELECT 1 FROM {g["name"]})' for g in zero_groups]
        relation_groups: list[dict] = list(multi_metas)
        residual_unary_views = 0
        residual_unary_guards = 0

        # A unary V_x that has been absorbed into a multi-frontier factor is
        # redundant at top level.  If no such factor contains x, keep V_x as an
        # answer relation or as an existential Boolean gate.
        for name, view in unary_view_by_var.items():
            if name in multi_vars:
                continue
            if name in answer_names:
                relation_groups.append(
                    {
                        'index': len(groups) + var_order.index(name),
                        'name': view,
                        'frontier': (name,),
                        'vars': {name},
                        'constants': 0,
                    }
                )
                residual_unary_views += 1
            else:
                guards.append(f'EXISTS (SELECT 1 FROM {view})')
                residual_unary_guards += 1

        global_joins = global_semis = global_crosses = 0
        bindings: dict[str, str] = {}
        from_sql = ''
        if relation_groups:
            ordered = self._group_order(relation_groups)
            first = ordered[0]
            first_alias = f'tg0'
            from_sql = f'{first["name"]} {first_alias}'
            bindings = {
                name: f'{first_alias}.{var_aliases[name]}'
                for name in first['frontier']
            }

            for pos, group in enumerate(ordered[1:], 1):
                alias = f'tg{pos}'
                shared = [name for name in group['frontier'] if name in bindings]
                new = [name for name in group['frontier'] if name not in bindings]
                if not group['vars']:
                    guards.append(f'EXISTS (SELECT 1 FROM {group["name"]})')
                    global_semis += 1
                    continue
                if shared and not new:
                    cond = ' AND '.join(
                        f'{alias}.{var_aliases[name]} = {bindings[name]}' for name in shared
                    )
                    guards.append(
                        f'EXISTS (SELECT 1 FROM {group["name"]} {alias} WHERE {cond})'
                    )
                    global_semis += 1
                    continue
                if shared:
                    cond = ' AND '.join(
                        f'{alias}.{var_aliases[name]} = {bindings[name]}' for name in shared
                    )
                    from_sql += f' INNER JOIN {group["name"]} {alias} ON {cond}'
                    global_joins += 1
                else:
                    from_sql += f' CROSS JOIN {group["name"]} {alias}'
                    global_crosses += 1
                for name in new:
                    bindings[name] = f'{alias}.{var_aliases[name]}'

        if projection:
            cols = []
            for i, var in enumerate(projection):
                expr = bindings.get(var.name)
                if expr is None:
                    raise ValueError(f'Unbound answer variable {var.name}')
                cols.append(f'{expr} AS a{i}')
            select = 'SELECT DISTINCT ' + ', '.join(cols)
        else:
            select = 'SELECT DISTINCT 1 AS one'

        with_sql = 'WITH ' + ', '.join(all_ctes) + ' '
        if from_sql:
            sql = with_sql + f'{select} FROM {from_sql}'
        else:
            sql = with_sql + select
        if guards:
            sql += ' WHERE ' + ' AND '.join(guards)

        return sql, {
            'atom_groups': len(groups),
            'candidate_branches': total_candidates,
            'active_candidate_branches': active_candidates,
            'static_empty_branches': static_empty,
            'frontier_variables_total': sum(len(f) for f in group_frontiers),
            'candidate_stages': candidate_stages,
            'candidate_inner_joins': candidate_joins,
            'candidate_semi_joins': candidate_semis,
            'candidate_cross_joins': candidate_crosses,
            'candidate_eliminated_vars': eliminated,
            'candidate_early_projections': early_projection,
            'candidate_absorbed_unary_filters': absorbed_filters,
            'unary_frontier_source_groups': sum(len(v) for v in unary_groups_by_var.values()),
            'unary_frontier_views': len(unary_view_by_var),
            'unary_frontier_consolidation_joins': unary_consolidation_joins,
            'absorbed_unary_views': len(absorbed_view_names),
            'unary_absorption_edges': unary_absorption_edges,
            'residual_unary_views': residual_unary_views,
            'residual_unary_guards': residual_unary_guards,
            'zero_frontier_groups': len(zero_groups),
            'top_level_groups_after_absorption': len(relation_groups),
            'global_inner_joins': global_joins,
            'global_semi_joins': global_semis,
            'global_cross_joins': global_crosses,
        }

