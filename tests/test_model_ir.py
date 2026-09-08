from src.core.model_ir import build_model_ir


def test_model_ir_extracts_components() -> None:
    markdown = """
# Model
## Sets and indices
- I: set of jobs
## Parameters
- p_i: processing time
## Decision Variables
- x_i: 1 if selected
## Objective Function
- Minimize total cost
## Constraints
- Capacity: sum p_i x_i <= B
## Assumptions
- Deterministic processing times
"""

    ir = build_model_ir(markdown, "demo")

    assert ir.problem_name == "demo"
    assert ir.objective_sense == "minimize"
    assert ir.sets[0].name == "I"
    assert ir.parameters[0].name == "p_i"
    assert ir.decision_variables[0].name == "x_i"
    assert ir.constraints


def test_model_ir_marks_ambiguous_objective_sense_unknown() -> None:
    ir = build_model_ir("## Objective\n- minimize or maximize F(x)")

    assert ir.objective_sense == "unknown"
