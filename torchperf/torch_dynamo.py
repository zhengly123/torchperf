import torch
from torch import _dynamo, fx
import traceback
from typing import List
from torch.fx import passes, symbolic_trace
import pydot


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


def draw_simple_graph(gm, fn):
    dot_graph = pydot.Dot("torchTX Graph", graph_type="graph")
    nodes: list[torch.fx.Node] = gm.graph.nodes
    for node in nodes:
        print(f"{node.name} {node}")
        label = f"{node.name}"
        if hasattr(node, "info"):
            label += f" | {node.info}"
        if hasattr(node, "ragged"):
            label += f" | {node.ragged}"
        try:
            if isinstance(
                node.meta["tensor_meta"], torch.fx.passes.shape_prop.TensorMetadata
            ):
                label += f" | {node.meta['tensor_meta'].shape}"
        except KeyError:
            None
        dot_graph.add_node(pydot.Node(str(node.name), label=label))
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                dot_graph.add_edge(pydot.Edge(arg.name, node.name))
            elif isinstance(arg, (list, tuple)):
                print("list of args", node.name)
                for v in arg:
                    if isinstance(v, torch.fx.node.Node):
                        dot_graph.add_edge(pydot.Edge(v.name, node.name))
    dot_graph.write_svg(fn)


def plot_graph_module(gm: torch.fx.GraphModule, fn):
    g = passes.graph_drawer.FxGraphDrawer(gm, "my_module")
    g.get_dot_graph().write_svg(fn)
    draw_simple_graph(gm, "simple_" + fn)


def plot_graph_module_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    plot_graph_module(gm, "a_dynamo_backend.svg")
    # serialization_backend(gm, example_inputs)

    def fake_run(*x):
        print("Nothing executed in the plot_graph_module_backend")
        return torch.randn([1, 2, 3])

    return lambda *x: gm


def get_dynamo_graph_modules(module, args, kwargs, full_graph=False):
    gms = []

    def backend_get_graph_module(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ):
        gms.append(gm)
        return gm.forward

    module = torch.compile(module, backend=backend_get_graph_module)
    module(*args, **kwargs)
    assert len(gms) > 0, "Does the graph hit cache?"
    if full_graph:
        assert len(gms) == 1, f"Captured {len(gms)} graphs"
    return gms
