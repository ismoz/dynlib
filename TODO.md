# TODO

This file lists the tasks and improvements planned for the project. Each task includes a brief description, status, and any relevant notes.

## Tasks

1. **Old `StructSpec` Behavioral Flags**
   - **Description**: Dropped `StructSpec` had these flags:
      - `use_history`
      - `use_f_history`
      - `dense_output`
      - `needs_jacobian`
      First two are not relevant anymore because steppers can declare any amount of history in 
      their workspaces. However, `dense_output` and `needs_jacobian` are still useful information. 
      Because you may add external Jacobian arg to sims and dense output needs a different pipeline 
      for results. Consider them placing into `StepperMeta` or `StepperSpec`.
   - **Status**: Planned.

2. **Multistep Methods**
   - **Description**: Implement multistep methods such as Adams and BDF.
   - **Status**: Planned.
   - **Notes**: Evaluate performance and accuracy trade-offs.

3. **Stiff Solvers**
   - **Description**: Add support for implicit methods and Jacobian computation for stiff systems.
   - **Status**: Planned.
   - **Notes**: Investigate efficient Jacobian computation strategies.

4. **Dense Output for Interpolation**
   - **Description**: Implement dense output for interpolation between grid points.
   - **Status**: Planned.
   - **Notes**: Test with various interpolation methods.

5. **CLI Tool for Command-Line Model Building**
   - **Description**: Develop a CLI tool to facilitate model building from the command line.
   - **Status**: Planned.
   - **Notes**: Ensure user-friendly interface and robust error handling.

6. **User Interrupt Handling**
   - **Description**: Implement support for handling user interrupts (e.g., Ctrl+C) gracefully.
   - **Status**: Planned.
   - **Notes**: Ensure proper cleanup and state preservation.
