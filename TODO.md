# TODO

This file lists the tasks and improvements planned for the project. Each task includes a brief description, status, and any relevant notes.

## Tasks

1. **Implement `StructSpec` Behavioral Flags**
   - **Description**: Implement behavioral flags in the `StructSpec` class to support advanced steppers and solvers.
   - **Status**: Planned.
   - **Sub-Tasks**:
     - **Flag: `use_history`**
       - **Description**: For multistep methods (like Adams-Bashforth) that need past solution points.
       - **Status**: Planned.
       - **Notes**: Ensure compatibility with existing solvers.
     - **Flag: `use_f_history`**
       - **Description**: For multistep methods that need past RHS evaluations.
       - **Status**: Planned.
       - **Notes**: Evaluate performance impact.
     - **Flag: `dense_output`**
       - **Description**: For continuous output interpolation between grid points.
       - **Status**: Planned.
       - **Notes**: Test with various interpolation methods.
     - **Flag: `needs_jacobian`**
       - **Description**: For implicit/stiff solvers.
       - **Status**: Planned.
       - **Notes**: Investigate Jacobian computation strategies.


3. **Multistep Methods**
   - **Description**: Implement multistep methods such as Adams and BDF.
   - **Status**: Planned.
   - **Notes**: Evaluate performance and accuracy trade-offs.

4. **Stiff Solvers**
   - **Description**: Add support for implicit methods and Jacobian computation for stiff systems.
   - **Status**: Planned.
   - **Notes**: Investigate efficient Jacobian computation strategies.

5. **Dense Output for Interpolation**
   - **Description**: Implement dense output for interpolation between grid points.
   - **Status**: Planned.
   - **Notes**: Test with various interpolation methods.

6. **CLI Tool for Command-Line Model Building**
   - **Description**: Develop a CLI tool to facilitate model building from the command line.
   - **Status**: Planned.
   - **Notes**: Ensure user-friendly interface and robust error handling.

8. **User Interrupt Handling**
   - **Description**: Implement support for handling user interrupts (e.g., Ctrl+C) gracefully.
   - **Status**: Planned.
   - **Notes**: Ensure proper cleanup and state preservation.
