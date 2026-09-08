from src.core.solver_adapter import SolverToolCapability, SolverToolReport, recommended_tools_for_profile


def test_recommended_tools_use_available_networkx() -> None:
    report = SolverToolReport(
        tools=[
            SolverToolCapability(
                name="NetworkX",
                package="networkx",
                status="available",
                use_cases=["graphs"],
                fallback="manual graph algorithms",
            )
        ]
    )

    assert recommended_tools_for_profile("network_analysis", report) == ["NetworkX"]


def test_recommended_tools_fallback_when_missing() -> None:
    report = SolverToolReport(
        tools=[
            SolverToolCapability(
                name="Pyomo",
                package="pyomo",
                status="missing",
                use_cases=["optimization"],
                fallback="template baseline",
            )
        ]
    )

    assert recommended_tools_for_profile("optimization", report) == ["template_baseline"]
