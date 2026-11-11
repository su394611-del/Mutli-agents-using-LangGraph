import argparse
import json
import operator
import os
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError


# === Common helper ============================================================
# Keep diagrams offline: Mermaid source is generated without making a web call.
# An online PNG renderer is NOT required to run any example.
def compile_graph(builder: StateGraph, diagram_name: str, export_dir: Path | None = None):
    graph = builder.compile()
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / f"{diagram_name}.mmd").write_text(
            graph.get_graph().draw_mermaid(), encoding="utf-8"
        )
    return graph


def _openai_llm(model_name: str | None = None):
    """Load OpenAI integration only when explicitly requested."""
    try:
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional packages: pip install langchain-openai python-dotenv"
        ) from exc
    load_dotenv()
    if not os.getenv("key_4LiqsgHst8glMMrY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in the environment or in a .env file."
        )
    return ChatOpenAI(model=model_name or os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))


# === 01. Introduction: single node, string state =============================
def example_greeting(export_dir: Path | None = None) -> dict:
    def fn_greeting(name: str) -> str:
        return f"Hello {name}"

    builder = StateGraph(str)
    builder.add_node("Greeting_Node", fn_greeting)
    builder.add_edge(START, "Greeting_Node")
    builder.add_edge("Greeting_Node", END)
    graph = compile_graph(builder, "01_greeting", export_dir)
    return {"output": graph.invoke("Sudhanshu")}


# === 02. Nodes: factorial =====================================================
def example_factorial(export_dir: Path | None = None) -> dict:
    def fn_factorial(n: int) -> int:
        if n < 0:
            raise ValueError("Factorial is undefined for negative integers")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    builder = StateGraph(int)
    builder.add_node("Factorial_Node", fn_factorial)
    builder.add_edge(START, "Factorial_Node")
    builder.add_edge("Factorial_Node", END)
    graph = compile_graph(builder, "02_factorial", export_dir)
    return {"input": 4, "factorial": graph.invoke(4)}


# === 03. Optional LLM node ====================================================
def example_llm_node(export_dir: Path | None = None, model_name: str | None = None) -> dict:
    llm = _openai_llm(model_name)

    def call_llm(prompt: str) -> str:
        response = llm.invoke(prompt)
        return str(response.content)

    builder = StateGraph(str)
    builder.add_node("LLM_Node", call_llm)
    builder.add_edge(START, "LLM_Node")
    builder.add_edge("LLM_Node", END)
    graph = compile_graph(builder, "03_llm_node", export_dir)
    return {"output": graph.invoke("Explain the concept of a black hole in one line.")}


# === 04. Normal edges: reverse then uppercase ================================
def example_normal_edges(export_dir: Path | None = None) -> dict:
    def reverse_string(text: str) -> str:
        return text[::-1]

    def to_uppercase(text: str) -> str:
        return text.upper()

    builder = StateGraph(str)
    builder.add_node("Reverse_Node", reverse_string)
    builder.add_node("Uppercase_Node", to_uppercase)
    builder.add_edge(START, "Reverse_Node")
    builder.add_edge("Reverse_Node", "Uppercase_Node")
    builder.add_edge("Uppercase_Node", END)
    graph = compile_graph(builder, "04_normal_edges", export_dir)
    return {"input": "LangGraph is Mutliagent ", "output": graph.invoke("LangGraph is easy")}


# === 05. Conditional edges ====================================================
def example_conditional_edges(export_dir: Path | None = None) -> dict:
    def add_ten(number: int) -> int:
        return number + 10

    def multiply_by_two(number: int) -> int:
        return number * 2

    def subtract_five(number: int) -> int:
        return number - 5

    def route_based_on_value(number: int) -> str:
        return "multiply" if number > 50 else "subtract"

    builder = StateGraph(int)
    builder.add_node("Add_Node", add_ten)
    builder.add_node("Multiply_Node", multiply_by_two)
    builder.add_node("Subtract_Node", subtract_five)
    builder.add_edge(START, "Add_Node")
    builder.add_conditional_edges("Add_Node", route_based_on_value,
                                  {"multiply": "Multiply_Node", "subtract": "Subtract_Node"})
    builder.add_edge("Multiply_Node", END)
    builder.add_edge("Subtract_Node", END)
    graph = compile_graph(builder, "05_conditional_edges", export_dir)
    return {"45": graph.invoke(45), "20": graph.invoke(20)}  # 110, 25


# === 06. TypedDict state schema ===============================================
def example_typed_dict_state(export_dir: Path | None = None) -> dict:
    class NumberState(TypedDict):
        num: int
        num_type: NotRequired[str]
        output: NotRequired[str]

    def check_even_odd(state: NumberState) -> dict:
        return {"num_type": "even" if state["num"] % 2 == 0 else "odd"}

    def handle_even(state: NumberState) -> dict:
        return {"output": f'{state["num"]} is even'}

    def handle_odd(state: NumberState) -> dict:
        return {"output": f'{state["num"]} is odd'}

    builder = StateGraph(NumberState)
    builder.add_node("Check_Node", check_even_odd)
    builder.add_node("Even_Node", handle_even)
    builder.add_node("Odd_Node", handle_odd)
    builder.add_edge(START, "Check_Node")
    builder.add_conditional_edges("Check_Node", lambda s: s["num_type"],
                                  {"even": "Even_Node", "odd": "Odd_Node"})
    builder.add_edge("Even_Node", END)
    builder.add_edge("Odd_Node", END)
    graph = compile_graph(builder, "06_typed_dict", export_dir)
    return {"even": graph.invoke({"num": 10}), "odd": graph.invoke({"num": 7})}


# === 07. Pydantic state and validation ========================================
def example_pydantic_state(export_dir: Path | None = None) -> dict:
    class GraphState(BaseModel):
        input: int = Field(ge=1, le=10)
        output: int | None = Field(default=None, ge=1, le=10)

    def calc_square(state: GraphState) -> dict:
        # Explicitly validate output. LangGraph's state schema validation focuses
        # on graph inputs and does not guarantee every partial node update is
        # checked as an output model on each step.
        squared = state.input * state.input
        validated = GraphState(input=state.input, output=squared)
        return {"output": validated.output}

    builder = StateGraph(GraphState)
    builder.add_node("square", calc_square)
    builder.add_edge(START, "square")
    builder.add_edge("square", END)
    graph = compile_graph(builder, "07_pydantic", export_dir)
    results = {"input_3": graph.invoke({"input": 3})}
    for invalid in (15, 8):
        try:
            graph.invoke({"input": invalid})
            results[f"input_{invalid}"] = "UNEXPECTEDLY ACCEPTED"
        except ValidationError as exc:
            results[f"input_{invalid}"] = {
                "expected_validation_error": True,
                "detail": exc.errors()[0]["msg"],
            }
    return results


# === 08. Separate input, internal and output schemas =========================
def example_input_output_schemas(export_dir: Path | None = None) -> dict:
    class InputState(TypedDict):
        input: str

    class GlobalState(TypedDict):
        character_count: int
        word_count: int

    class OutputState(TypedDict):
        output: str

    def analyze_text(state: InputState) -> GlobalState:
        return {"character_count": len(state["input"]),
                "word_count": len(state["input"].split())}

    def output_msg(state: GlobalState) -> OutputState:
        long = state["character_count"] > 20 or state["word_count"] > 3
        return {"output": "Input is long" if long else "Input is short"}

    builder = StateGraph(state_schema=GlobalState, input_schema=InputState,
                         output_schema=OutputState)
    builder.add_node("analyze_text", analyze_text)
    builder.add_node("output_msg", output_msg)
    builder.add_edge(START, "analyze_text")
    builder.add_edge("analyze_text", "output_msg")
    builder.add_edge("output_msg", END)
    graph = compile_graph(builder, "08_input_output_schemas", export_dir)
    return {"output": graph.invoke({"input": "Hello World, How Are You?"})}


# === 09. Private (intermediate) schema ========================================
def example_private_schema(export_dir: Path | None = None) -> dict:
    class GlobalState(TypedDict):
        input: str
        output: NotRequired[str]

    class PrivateState(TypedDict):
        character_count: int
        word_count: int

    def analyze_text(state: GlobalState) -> PrivateState:
        return {"character_count": len(state["input"]),
                "word_count": len(state["input"].split())}

    def output_msg(state: PrivateState) -> dict:
        long = state["character_count"] > 20 or state["word_count"] > 3
        return {"output": "Input is long" if long else "Input is short"}

    builder = StateGraph(GlobalState)
    builder.add_node("analyze_text", analyze_text)
    builder.add_node("output_msg", output_msg)
    builder.add_edge(START, "analyze_text")
    builder.add_edge("analyze_text", "output_msg")
    builder.add_edge("output_msg", END)
    graph = compile_graph(builder, "09_private_schema", export_dir)
    return {"output": graph.invoke({"input": "Hello World!"})}


# === 10. Reducers: default overwrite =========================================
def example_reducer_overwrite(export_dir: Path | None = None) -> dict:
    class CounterState(TypedDict):
        counter: int

    def node1(state: CounterState) -> dict:
        return {"counter": 1}

    def node2(state: CounterState) -> dict:
        return {"counter": 2}

    builder = StateGraph(CounterState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)
    graph = compile_graph(builder, "10_reducer_overwrite", export_dir)
    return graph.invoke({"counter": 0})


# === 11. Reducers: operator.add with numeric values ==========================
def example_reducer_numeric_add(export_dir: Path | None = None) -> dict:
    class CounterState(TypedDict):
        counter: Annotated[int, operator.add]

    def node1(state: CounterState) -> dict:
        return {"counter": 1}

    def node2(state: CounterState) -> dict:
        return {"counter": 2}

    builder = StateGraph(CounterState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)
    graph = compile_graph(builder, "11_reducer_numeric", export_dir)
    return graph.invoke({"counter": 0})


# === 12. Reducers: append lists with operator.add =============================
def example_reducer_list_append(export_dir: Path | None = None) -> dict:
    class CounterState(TypedDict):
        counter: Annotated[list[int], operator.add]

    def node1(state: CounterState) -> dict:
        return {"counter": [1]}

    def node2(state: CounterState) -> dict:
        return {"counter": [2]}

    builder = StateGraph(CounterState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)
    graph = compile_graph(builder, "12_reducer_list", export_dir)
    return graph.invoke({"counter": []})


# === 13. Reducers: custom ordered, unique list merge =========================
def example_custom_reducer(export_dir: Path | None = None) -> dict:
    def merge_unique(existing: list[int], new: list[int]) -> list[int]:
        result = list(existing)
        for item in new:
            if item not in result:
                result.append(item)
        return result

    class CounterState(TypedDict):
        counter: Annotated[list[int], merge_unique]

    def node1(state: CounterState) -> dict:
        return {"counter": [1, 2]}

    def node2(state: CounterState) -> dict:
        return {"counter": [2, 3]}

    builder = StateGraph(CounterState)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)
    graph = compile_graph(builder, "13_custom_reducer", export_dir)
    return graph.invoke({"counter": []})


# === 14. Built-in add_messages reducer =======================================
def example_add_messages(export_dir: Path | None = None) -> dict:
    # The reducer merges messages by stable ID (unlike naive list concatenation).
    existing = [AIMessage(content="Hello! How can I help you?", id="ai-1"),
                HumanMessage(content="Can you help me in Python?", id="human-1")]
    additional = AIMessage(content="Sure; what Python problem are you solving?", id="ai-2")
    merged = add_messages(existing, [additional])
    edited = add_messages(merged, [AIMessage(content="Absolutely!", id="ai-2")])
    return {"merged_count": len(merged),
            "updated_count": len(edited),
            "updated_last_message": edited[-1].content}


# === 15. MessagesState graph (offline assistant response) =====================
def example_messages_state(export_dir: Path | None = None) -> dict:
    class ChatMessageState(MessagesState):
        user_name: str

    def assistant_node(state: ChatMessageState) -> dict:
        question = state["messages"][-1].content
        return {"messages": [AIMessage(content=(
            f'Hello {state["user_name"]}! Your question was: {question}. '
            "Fun bird fact: hummingbirds can fly backward."
        ))]}

    builder = StateGraph(ChatMessageState)
    builder.add_node("assistant_node", assistant_node)
    builder.add_edge(START, "assistant_node")
    builder.add_edge("assistant_node", END)
    graph = compile_graph(builder, "15_messages_state", export_dir)
    output = graph.invoke({"user_name": "Sudhanshu", "messages": [
        HumanMessage(content="Tell me a bird fact in one line.")
    ]})
    return {"user_name": output["user_name"],
            "conversation": [m.content for m in output["messages"]]}


# === 16. MessagesState graph with real OpenAI LLM (optional) ==================
def example_messages_state_llm(export_dir: Path | None = None,
                               model_name: str | None = None) -> dict:
    llm = _openai_llm(model_name)

    class ChatMessageState(MessagesState):
        user_name: str

    def llm_node(state: ChatMessageState) -> dict:
        user_question = str(state["messages"][-1].content)
        prompt = [HumanMessage(content=(
            f'Give a brief personalized response to {state["user_name"]}. '
            f'Question: {user_question}'
        ))]
        response = llm.invoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(ChatMessageState)
    builder.add_node("llm_node", llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    graph = compile_graph(builder, "16_messages_state_llm", export_dir)
    output = graph.invoke({"user_name": "Sudhanshu", "messages": [
        HumanMessage(content="Tell me a fun bird fact in one line.")
    ]})
    return {"conversation": [str(m.content) for m in output["messages"]]}


# === 17. Subgraph invoked inside a parent node (different state schemas) ======
def example_subgraph_wrapper(export_dir: Path | None = None) -> dict:
    class ITState(TypedDict):
        new_hire_name: str
        assets: list[str]

    def create_email(state: ITState) -> dict:
        email = f'{state["new_hire_name"].lower().replace(" ", ".")}@company.com'
        return {"assets": state.get("assets", []) + [email]}

    def assign_laptop(state: ITState) -> dict:
        return {"assets": state.get("assets", []) + ["MacBook Pro"]}

    it_builder = StateGraph(ITState)
    it_builder.add_node("Create_Email_Node", create_email)
    it_builder.add_node("Assign_Laptop_Node", assign_laptop)
    it_builder.add_edge(START, "Create_Email_Node")
    it_builder.add_edge("Create_Email_Node", "Assign_Laptop_Node")
    it_builder.add_edge("Assign_Laptop_Node", END)
    it_subgraph = compile_graph(it_builder, "17_it_subgraph_wrapper", export_dir)

    class HRState(TypedDict):
        employee_name: str
        hr_status: NotRequired[str]

    def hr_welcome(state: HRState) -> dict:
        return {"hr_status": f'HR Notes: Welcome package sent to {state["employee_name"]}.'}

    def call_it_department(state: HRState) -> dict:
        it_output = it_subgraph.invoke({"new_hire_name": state["employee_name"],
                                        "assets": []})
        assets_str = ", ".join(it_output["assets"])
        return {"hr_status": f'{state.get("hr_status", "")}\nIT Notes: {assets_str} assigned.'}

    hr_builder = StateGraph(HRState)
    hr_builder.add_node("HR_Welcome_Node", hr_welcome)
    hr_builder.add_node("IT_Wrapper_Node", call_it_department)
    hr_builder.add_edge(START, "HR_Welcome_Node")
    hr_builder.add_edge("HR_Welcome_Node", "IT_Wrapper_Node")
    hr_builder.add_edge("IT_Wrapper_Node", END)
    parent_graph = compile_graph(hr_builder, "17_hr_parent_wrapper", export_dir)
    return parent_graph.invoke({"employee_name": "Alice Smith"})


# === 18. Add compiled subgraph directly as node (shared schema) ===============
def example_subgraph_direct(export_dir: Path | None = None) -> dict:
    class CompanyState(TypedDict):
        employee_name: str
        hr_notes: NotRequired[str]
        it_notes: NotRequired[str]

    def create_email(state: CompanyState) -> dict:
        email = f'{state["employee_name"].lower().replace(" ", ".")}@company.com'
        return {"it_notes": f"Email created: {email}."}

    def assign_laptop(state: CompanyState) -> dict:
        return {"it_notes": f'{state.get("it_notes", "")} MacBook Pro assigned.'.strip()}

    it_builder = StateGraph(CompanyState)
    it_builder.add_node("Create_Email_Node", create_email)
    it_builder.add_node("Assign_Laptop_Node", assign_laptop)
    it_builder.add_edge(START, "Create_Email_Node")
    it_builder.add_edge("Create_Email_Node", "Assign_Laptop_Node")
    it_builder.add_edge("Assign_Laptop_Node", END)
    it_subgraph = compile_graph(it_builder, "18_it_subgraph_direct", export_dir)

    def hr_welcome(state: CompanyState) -> dict:
        return {"hr_notes": f'Welcome package sent to {state["employee_name"]}.'}

    hr_builder = StateGraph(CompanyState)
    hr_builder.add_node("HR_Welcome_Node", hr_welcome)
    hr_builder.add_node("IT_Subgraph_Node", it_subgraph)
    hr_builder.add_edge(START, "HR_Welcome_Node")
    hr_builder.add_edge("HR_Welcome_Node", "IT_Subgraph_Node")
    hr_builder.add_edge("IT_Subgraph_Node", END)
    parent_graph = compile_graph(hr_builder, "18_hr_parent_direct", export_dir)
    return parent_graph.invoke({"employee_name": "Bob Jones"})


# === CLI and smoke-test ========================================================
EXAMPLES = {
    "greeting": (example_greeting, False),
    "factorial": (example_factorial, False),
    "llm_node": (example_llm_node, True),
    "normal_edges": (example_normal_edges, False),
    "conditional_edges": (example_conditional_edges, False),
    "typed_dict_state": (example_typed_dict_state, False),
    "pydantic_state": (example_pydantic_state, False),
    "input_output_schemas": (example_input_output_schemas, False),
    "private_schema": (example_private_schema, False),
    "reducer_overwrite": (example_reducer_overwrite, False),
    "reducer_numeric_add": (example_reducer_numeric_add, False),
    "reducer_list_append": (example_reducer_list_append, False),
    "custom_reducer": (example_custom_reducer, False),
    "add_messages": (example_add_messages, False),
    "messages_state": (example_messages_state, False),
    "messages_state_llm": (example_messages_state_llm, True),
    "subgraph_wrapper": (example_subgraph_wrapper, False),
    "subgraph_direct": (example_subgraph_direct, False),
}


def self_test(results: dict) -> None:
    """Assert outputs from the deterministic, API-free teaching examples."""
    def check(name, actual, expected):
        assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"

    check("greeting", results["greeting"]["output"], "Hello Sudhanshu")
    check("factorial", results["factorial"]["factorial"], 24)
    check("normal_edges", results["normal_edges"]["output"], "YSAE SI HPARGGNAL")
    check("conditional_edges/45", results["conditional_edges"]["45"], 110)
    check("conditional_edges/20", results["conditional_edges"]["20"], 25)
    check("typed_dict/even", results["typed_dict_state"]["even"]["output"], "10 is even")
    check("typed_dict/odd", results["typed_dict_state"]["odd"]["output"], "7 is odd")
    check("pydantic/3", results["pydantic_state"]["input_3"]["output"], 9)
    assert all(results["pydantic_state"][f"input_{i}"]["expected_validation_error"]
               for i in (15, 8))
    check("input_output", results["input_output_schemas"]["output"]["output"], "Input is long")
    check("private", results["private_schema"]["output"]["output"], "Input is short")
    check("overwrite", results["reducer_overwrite"]["counter"], 2)
    check("numeric_add", results["reducer_numeric_add"]["counter"], 3)
    check("list_append", results["reducer_list_append"]["counter"], [1, 2])
    check("custom_reducer", results["custom_reducer"]["counter"], [1, 2, 3])
    check("add_messages/update", results["add_messages"]["updated_last_message"], "Absolutely!")
    assert len(results["messages_state"]["conversation"]) == 2
    assert "alice.smith@company.com" in results["subgraph_wrapper"]["hr_status"]
    assert "bob.jones@company.com" in results["subgraph_direct"]["it_notes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph Chapter 10 — complete collection of runnable examples",
        allow_abbrev=False,
    )
    parser.add_argument("--list", action="store_true", help="Show available topic names")
    parser.add_argument("--topic", choices=list(EXAMPLES),
                        help="Run only one example (default: all API-free examples)")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run OpenAI examples; requires OPENAI_API_KEY, incurs API charges")
    parser.add_argument("--model", default=None, help="OpenAI model name, e.g. gpt-5.4-mini")
    parser.add_argument("--export-mermaid", type=Path, default=None,
                        help="Write graph diagrams as .mmd Mermaid files (offline)")
    parser.add_argument("--json-output", type=Path, default=None,
                        help="Save complete outputs as JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="Run all deterministic topics and validate expected results")
    args = parser.parse_args(argv)
    if args.list:
        for name, (_, needs_llm) in EXAMPLES.items():
            print(f'{name:<24} {"[OPENAI API]" if needs_llm else "[offline]"}')
        return 0

    if args.self_test and args.topic:
        parser.error("--self-test cannot be combined with --topic")
    if args.self_test and args.with_llm:
        parser.error("--self-test does not call remote APIs; remove --with-llm")

    selected = [args.topic] if args.topic else [
        name for name, (_, requires_llm) in EXAMPLES.items()
        if args.with_llm or not requires_llm
    ]
    outputs = {}
    for name in selected:
        func, needs_llm = EXAMPLES[name]
        print(f"\n{'=' * 12} {name} {'=' * 12}")
        kwargs = {"export_dir": args.export_mermaid}
        if needs_llm:
            kwargs["model_name"] = args.model
        try:
            result = func(**kwargs)
        except Exception as exc:
            print(f"ERROR in {name}: {type(exc).__name__}: {exc}")
            raise
        outputs[name] = result
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if args.self_test:
        self_test(outputs)
        print(f"\nSELF-TEST PASSED: {len(outputs)} offline examples")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(outputs, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Saved results to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
