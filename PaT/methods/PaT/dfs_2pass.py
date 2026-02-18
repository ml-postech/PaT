import dataclasses

from ...langrt import LrtFunctionDef, LrtNode, LrtProgram, SymbolName
from ..shared import CodeGenContext, CodeGenJournal, CodeGenJournalist
from .gen_once import GenOnceSig
from .runner import runner_evaluate_cases

@dataclasses.dataclass
class Node:
    ancestor_nodes: list[LrtNode]
    ancestor_funcs: list[SymbolName]
    cur: SymbolName
    siblings: list[LrtNode]
    children: list["Node"]
    pass


class PaTDfs2Pass:
    def __init__(
        self,
        ctx: CodeGenContext,
        opt_max_depth: int,
        opt_refine_leaf: bool,
        opt_patch_refine_root_docstring: bool,
        gen_pass_1: GenOnceSig,
        gen_pass_2: GenOnceSig,
        ancestors: list[LrtNode],
        func: LrtFunctionDef,
        descendants: list[LrtNode],
    ):
        self.ctx = ctx
        self.opt_max_depth = opt_max_depth
        self.opt_refine_leaf = opt_refine_leaf
        self.opt_patch_refine_root_docstring = opt_patch_refine_root_docstring
        self.gen_pass_1 = gen_pass_1
        self.gen_pass_2 = gen_pass_2

        self.funcs: dict[SymbolName, LrtFunctionDef] = {}
        self.vis: dict[SymbolName, bool] = {}
        self.shared_descendant_nodes: list[LrtNode] = []
        self.shared_descendant_funcs: list[SymbolName] = []

        for n in ancestors:
            if n.kind == "function":
                self.funcs[n.name] = n
                self.vis[n.name] = True
            else:
                self.shared_descendant_nodes.append(n)
        for n in descendants:
            if n.kind == "function":
                self.funcs[n.name] = n
                self.vis[n.name] = True
                self.shared_descendant_funcs.append(n.name)
            else:
                self.shared_descendant_nodes.append(n)
        self.funcs[func.name] = func
        self.vis[func.name] = False

        self.root = Node(
            ancestor_nodes=[n for n in ancestors if n.kind != "function"],
            ancestor_funcs=[n.name for n in ancestors if n.kind == "function"],
            cur=func.name,
            siblings=[],
            children=[],
        )
        self._ancestors = ancestors
        self._func = func
        self._descendants = descendants

    async def run(self) -> tuple[tuple[LrtFunctionDef, list[LrtNode]] | None, CodeGenJournal]:
        self.ctx.log.in_scope(f"PaT_dfs_2pass[{self._func.name}(...)]")
        _sj = CodeGenJournalist(
            self.ctx,
            "PaT_dfs_2pass",
            (self._ancestors, self._func, self._descendants),
        )

        root, _sj_ch = await self.dfs(self.root, 0)
        _sj.append(_sj_ch)
        if root is None:
            _err = "failed to generate program"
            self.ctx.log.warn(_err)
            return None, _sj.collect_err(error=_err)

        # verify results
        program = self._subtree_of(root)
        program = self.ctx.lrt.prettify(LrtProgram(module=(), nodes=program))
        if not (ret_func := program.find(LrtFunctionDef, self._func.name)):
            _err = "requested function not found in generated program"
            self.ctx.log.warn(_err)
            return None, _sj.collect_err(error=_err)
        ret_nodes = program.excluding(ret_func)

        return (ret_func, ret_nodes), _sj.collect_gen((ret_func, ret_nodes))

    async def dfs(self, node: Node, depth: int) -> tuple[Node | None, CodeGenJournal | None]:
        if self.vis.get(node.cur, False):
            return node, None
        self.vis[node.cur] = True
        _sj = CodeGenJournalist(
            self.ctx,
            "PaT_dfs_2pass[dfs]",
            (self._ancestors_of(node), self.funcs[node.cur], self._descendants_of(node)),
        )
        if depth >= self.opt_max_depth:
            return None, _sj.collect_err("max depth reached")

        fn_current_node_initial = self.funcs[node.cur]
        
        retry_1 = 3
        
        for i in range(retry_1):
            self.ctx.log.string(f"Strategy {i+1}: Attempting direct implementation (Pass 2) for {fn_current_node_initial.name}...")
            
            ret_2_result, _sj_2, tests = await self.gen_pass_2(
                self.ctx, self._ancestors_of(node), fn_current_node_initial, node.siblings + self._descendants_of(node)
            )
            _sj.append(_sj_2)
            
            if not tests:
                continue
            else:
                break
        if ret_2_result is not None:
            final_func_impl, associated_nodes_from_runner = ret_2_result

            if final_func_impl: 
                self.funcs[final_func_impl.name] = final_func_impl 

                self.ctx.log.string(f"Direct implementation (Pass 2) succeeded for {final_func_impl.name}. Returning this solution.")
                
                solved_node = Node(
                    ancestor_nodes=node.ancestor_nodes,
                    ancestor_funcs=node.ancestor_funcs,
                    cur=final_func_impl.name,
                    siblings=associated_nodes_from_runner,
                    children=[]
                )
                return solved_node, _sj.collect_gen((final_func_impl, associated_nodes_from_runner))

        
        self.ctx.log.warn(f"Direct implementation (Pass 2) failed for {fn_current_node_initial.name} (heuristic detected failure). Switching to decomposition (Pass 1).")

        retry_2 = 5
        
        prev_pass_count = -1
        prev_fn = None
        for re in range(retry_2):
            self.ctx.log.warn(f"Division try: {re+1}")

            fn_for_pass1 = fn_current_node_initial
            if self.opt_patch_refine_root_docstring and depth == 0:
                fn_for_pass1 = fn_for_pass1.model_copy()
                fn_for_pass1.docstring = fn_current_node_initial.docstring
                fn_for_pass1 = self.ctx.lrt.prettify(fn_for_pass1)

            ret_1_result, _sj_1_fallback = await self.gen_pass_1(
                self.ctx, self._ancestors_of(node), fn_for_pass1, node.siblings + self._descendants_of(node)
            )
            _sj.append(_sj_1_fallback)

            if ret_1_result is None:
                self.ctx.log.error(f"Decomposition (Pass 1) also failed for {fn_current_node_initial.name}. Cannot proceed.")
                continue

            fn_after_divide, children_from_divide = ret_1_result
            self.funcs[fn_after_divide.name] = fn_after_divide

            self.ctx.log.string(f"Decomposition (Pass 1) succeeded for {fn_after_divide.name}. Recursing into {len(children_from_divide)} children.")

            processed_children_nodes = []
            for child in children_from_divide:
                if child.kind != "function":
                    node.siblings.append(child)
                    continue
                if child.name in self.vis:
                    continue
                self.funcs[child.name] = child
                
                ch_node = Node(
                    ancestor_nodes=node.ancestor_nodes + node.siblings,
                    ancestor_funcs=node.ancestor_funcs + [node.cur],
                    cur=child.name,
                    siblings=[],
                    children=[],
                )
                ch_node_result, _sj_ch = await self.dfs(ch_node, depth + 1)
                _sj.append(_sj_ch)
                if ch_node_result is not None:
                    processed_children_nodes.append(ch_node_result)
            
            node.children.extend(processed_children_nodes)

            final_log_result_func = self.funcs[fn_after_divide.name] 
            final_log_result_children = sum([self._subtree_of(ch) for ch in node.children], [])

            log = _sj.collect_gen((final_log_result_func, final_log_result_children))
            
            if not tests:
                self.ctx.log.warn(f"no tests available for check")
                return node, log
            
            results = await runner_evaluate_cases(
                ctx=self.ctx,
                cfg_timeout=5.0,
                impls=[(LrtProgram(module = (), nodes=[final_log_result_func]+final_log_result_children), final_log_result_func)],
                tests = tests,
                )
            if not results or not results[0]:
                self.ctx.log.string("No test results found for the generated function.")
                continue
            
            single_impl_test_results = results[0]
            
            curr_pass = sum(1 for case_result in single_impl_test_results if case_result.ok)
            
            all_tests_passed = all(case_result.ok for case_result in single_impl_test_results)
            
            if all_tests_passed:
                self.ctx.log.string("All tests passed for the generated function.")
                break
            elif curr_pass <= prev_pass_count and curr_pass > 0:
                self.ctx.log.string("The number of pass is not increased.")
                self.funcs[prev_fn.name] = prev_fn
                break   
            else:
                prev_fn = fn_after_divide
                prev_pass_count = curr_pass
                self.ctx.log.string("Generated function failed some tests.")
                for i, case_result in enumerate(single_impl_test_results):
                    if not case_result.ok:
                        print(f"Test case {i+1} failed. Error: {case_result.error}") 
                continue
            
        return node, log
        
    def _ancestors_of(self, node: Node) -> list[LrtNode]:
        ret = [self.funcs[n] for n in node.ancestor_funcs] + node.ancestor_nodes
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    def _descendants_of(self, node: Node) -> list[LrtNode]:
        # all descendants of a node, excluding itself
        def __recurse(p: Node) -> list[LrtNode]:
            ret = []
            for ch in p.children:
                ret.append(self.funcs[ch.cur])
                ret.extend(ch.siblings)
                ret += __recurse(ch)
            return ret

        ret = __recurse(node)
        ret += self.shared_descendant_nodes
        ret += [self.funcs[n] for n in self.shared_descendant_funcs]
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    def _subtree_of(self, node: Node) -> list[LrtNode]:
        ret: list[LrtNode] = []
        ret.append(self.funcs[node.cur])
        ret.extend(node.siblings)
        for child in node.children:
            ret += self._subtree_of(child)
        return self.ctx.lrt._parse.deduplicate_nodes(ret)

    pass
