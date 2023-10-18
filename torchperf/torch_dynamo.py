import torch
from torch import _dynamo, fx
import traceback
from typing import List


def explain(compiled_func, *args, **kwargs):
    result = torch._dynamo.explain(compiled_func)(*args, **kwargs)

    if isinstance(result, tuple):  # Torch 2.0
        (
            explanation,
            out_guards,
            graphs,
            ops_per_graph,
            break_reasons,
            explanation_verbose,
        ) = result
        print(f"Explanation_verbose: {explanation_verbose}")
        # print(f'Generating {len(graphs)} graphs')
        for i, (graph_guard, graph, ops, break_reason) in enumerate(
            zip(out_guards, graphs, ops_per_graph, break_reasons)
        ):
            print("GRAPH", i)
            print("++graph_guard:", len(graph_guard))
            for guard in graph_guard:
                print(guard)
            print("++graph:")
            print(graph.print_readable(print_output=False))
            print("++ops:", len(ops))
            for op in ops:
                print(op)
            print("++break_reason:", break_reason.reason)
            print("".join(traceback.format_list(break_reason.user_stack)))
        print("finish")
    else:  # Torch 2.1
        print(result)
        for i, graph in enumerate(result.graphs):
            print(f"++Graph {i}:")
            print(graph.print_readable(print_output=False))


def serialization_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    print("=" * 20)
    print(gm.graph)
    print("=" * 20)
    print(gm.code)
    print("=" * 20)
    gm.print_readable()
    print("=" * 20)
    print(example_inputs)
    print("=" * 20)
    gm.graph.print_tabular()
    gm.to_folder("foo_full", "UNet")
    return gm.forward
