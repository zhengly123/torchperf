import torch
from torch import _dynamo, fx
import traceback
from typing import List, Callable
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


def draw_simple_graph(
    gm, fn, *, func: Callable[[torch.fx.Node], str] = None, limit_node_num=100
):
    dot_graph = pydot.Dot("torchTX Graph", graph_type="graph")
    nodes: list[torch.fx.Node] = gm.graph.nodes
    node_names = set() # To limit the number of nodes
    for node in nodes:
        if limit_node_num is not None and len(node_names) > limit_node_num:
            break
        node_names.add(node.name)
        label = f"{node.name} ({node.op})"
        node_style = {
            "shape": "record",
            "fillcolor": "white",
            "style": "rounded, filled",
        }
        if hasattr(node, "info"):
            label += f"\\n| {node.info} "
            if hasattr(node.info, "isRagged") and node.info.isRagged():
                node_style["fillcolor"] = "#9eb8d9"
            if hasattr(node.info, "is_shape") and node.info.is_shape:
                node_style["fillcolor"] = "#40e0d0"
        if hasattr(node, "ragged"):
            label += f"\\n| {node.ragged}"
        try:
            if isinstance(
                node.meta["tensor_meta"], torch.fx.passes.shape_prop.TensorMetadata
            ):
                label += f"\\n| {node.meta['tensor_meta'].shape}"
        except KeyError:
            None
        if func is not None: # User defined label
            label += "\n" + func(node)
        dot_graph.add_node(
            # "{}" makes the label in horizontal blocks
            pydot.Node(str(node.name), label="{" + label + "}", **node_style)
        )
    for node in nodes:
        if limit_node_num is not None and node.name not in node_names:
            continue
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                dot_graph.add_edge(pydot.Edge(arg.name, node.name))
            elif isinstance(arg, (list, tuple)):
                for v in arg:
                    if isinstance(v, torch.fx.node.Node):
                        dot_graph.add_edge(pydot.Edge(v.name, node.name))
    print(f"[INFO] FX graph written to {fn}")
    dot_graph.write_svg(fn)


def plot_graph_module(
    gm: torch.fx.GraphModule,
    fn: str,
    *,
    func: Callable[[torch.fx.Node], str] = None,
    limit_node_num=100,
):
    g = passes.graph_drawer.FxGraphDrawer(gm, "my_module")
    if not fn.endswith(".svg"):
        fn += ".svg"
    g.get_dot_graph().write_svg(fn)
    draw_simple_graph(gm, "simple_" + fn, func=func)


def plot_graph_module_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    plot_graph_module(gm, "a_dynamo_backend.svg")
    # serialization_backend(gm, example_inputs)

    def fake_run(*x):
        print("Nothing executed in the plot_graph_module_backend")
        return torch.randn([1, 2, 3])

    return lambda *x: gm


def get_dynamo_graph_modules_and_args(
    module, args, kwargs, full_graph=False, dynamic=None
):
    gms = []
    example_input_lists = []

    def backend_get_graph_module(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor | torch.SymInt]
    ):
        gms.append(gm)
        example_input_lists.append(example_inputs)
        # print("example_inputs", example_inputs)
        return gm.forward

    module = torch.compile(module, backend=backend_get_graph_module, dynamic=dynamic)
    module(*args, **kwargs)
    assert len(gms) > 0, "Does the graph hit cache?"
    if full_graph:
        assert len(gms) == 1, f"Captured {len(gms)} graphs"
    return gms, example_input_lists


def get_dynamo_graph_modules(module, args, kwargs, full_graph=False, dynamic=None):
    return get_dynamo_graph_modules_and_args(module, args, kwargs, full_graph, dynamic)[
        0
    ]
