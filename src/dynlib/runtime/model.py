# src/dynlib/runtime/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict
from pathlib import Path
import numpy as np

from dynlib.dsl.spec import ModelSpec
from dynlib.steppers.base import StructSpec

__all__ = ["Model"]

@dataclass(frozen=True)
class Model:
    """
    Compiled model ready for execution.
    
    Attributes:
        spec: Original ModelSpec
        stepper_name: Name of the stepper used
        struct: StructSpec from the stepper
        rhs: Compiled RHS callable
        events_pre: Compiled pre-events callable
        events_post: Compiled post-events callable
        stepper: Compiled stepper callable
        runner: Compiled runner callable
        spec_hash: Content hash of the spec
        dtype: NumPy dtype for the model
        rhs_source: Python source code for RHS function (if available)
        events_pre_source: Python source code for pre-events function (if available)
        events_post_source: Python source code for post-events function (if available)
        stepper_source: Python source code for stepper function (if available)
    """
    spec: ModelSpec
    stepper_name: str
    struct: StructSpec
    rhs: Callable
    events_pre: Callable
    events_post: Callable
    stepper: Callable
    runner: Callable
    spec_hash: str
    dtype: np.dtype
    rhs_source: Optional[str] = None
    events_pre_source: Optional[str] = None
    events_post_source: Optional[str] = None
    stepper_source: Optional[str] = None
    
    def export_sources(self, output_dir: str | Path) -> Dict[str, Path]:
        """
        Export all source code files to a directory for inspection.
        
        Args:
            output_dir: Directory path where source files will be written
            
        Returns:
            Dictionary mapping component names to their file paths
            
        Example:
            >>> model = build("decay.toml", stepper="euler")
            >>> files = model.export_sources("./compiled_sources")
            >>> print(files)
            {'rhs': Path('./compiled_sources/rhs.py'), 
             'events_pre': Path('./compiled_sources/events_pre.py'), ...}
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        
        # Export each component's source if available
        components = [
            ("rhs", self.rhs_source),
            ("events_pre", self.events_pre_source),
            ("events_post", self.events_post_source),
            ("stepper", self.stepper_source),
        ]
        
        for name, source in components:
            if source is not None:
                file_path = output_path / f"{name}.py"
                file_path.write_text(source, encoding="utf-8")
                exported[name] = file_path
        
        return exported

