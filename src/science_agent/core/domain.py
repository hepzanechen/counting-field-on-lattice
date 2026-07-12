from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScienceDomain(Protocol):
    name: str

    @property
    def observables(self) -> tuple[str, ...]: ...

    def catalog_for_prompt(self) -> str: ...

    def validate_parameters(
        self, model_name: str, proposed: dict[str, Any]
    ) -> dict[str, float]: ...

    def build_system(
        self,
        model_name: str,
        params: dict[str, Any],
        **options: Any,
    ) -> tuple[Any, Any, Any]: ...
