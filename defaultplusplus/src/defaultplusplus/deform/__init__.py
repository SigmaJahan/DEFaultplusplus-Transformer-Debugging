"""DEForm: transformer fault-injection mutation engine.

Three pieces:

  operators        Catalog of mutation operators with three-letter IDs,
                   parameter grammar (B / EU / EL search types), and the
                   fault component they target.
  injection        Static and dynamic injection mechanisms. A
                   FaultInjector is a context manager that mutates the
                   model on enter and restores it on exit, so paired
                   clean / faulty runs can share one model object across
                   seeds.
  validation       Mutation-killing logic: structural verifier and the
                   exact one-sided sign-flip permutation test over five
                   matched seeds (smallest n that admits an exact
                   one-sided test at alpha = 0.05; minimum p-value
                   1 / 2^5 ≈ 0.031).

The top-level :class:`FaultConfiguration` bundles a fault specification
``(model, task, unit, category, variant, layers, severity)``. The
:class:`Mutant` class is the per-config result: the killed flag, the
permutation-test p-value, and the labeled feature instance produced from
the paired clean and faulty fine-tuning traces.
"""

from .operators import (
    DECODER_ONLY_COMPONENTS,
    OPERATORS,
    Operator,
    OperatorComponent,
    OperatorSearchType,
    list_operators,
    operators_for_component,
    root_cause_label_space,
)
from .fault_config import FaultConfiguration, Mutant
from .injection import FaultInjector, StaticFault, DynamicFault
from .operator_impls import (
    get_expected_modules,
    get_expected_parameter_names,
    get_injector,
)
from .validation import (
    StructuralVerifier,
    sign_flip_permutation_test,
    is_killed,
)

__all__ = [
    "OPERATORS",
    "Operator",
    "OperatorComponent",
    "OperatorSearchType",
    "list_operators",
    "operators_for_component",
    "root_cause_label_space",
    "DECODER_ONLY_COMPONENTS",
    "FaultConfiguration",
    "Mutant",
    "FaultInjector",
    "StaticFault",
    "DynamicFault",
    "get_injector",
    "get_expected_parameter_names",
    "get_expected_modules",
    "StructuralVerifier",
    "sign_flip_permutation_test",
    "is_killed",
]
