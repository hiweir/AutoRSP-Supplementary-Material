# Appendix: End-to-End Execution Example of AutoRSP

This appendix presents a **fully reproducible end-to-end execution example** of AutoRSP, illustrating the complete pipeline from raw robotic code input to a verified Robot Skill Primitive (RSP) class output. The example uses the *turtlebot3_example* dataset (consistent with the main paper) and includes runtime logs, workflow visualizations, and intermediate outputs to ensure full transparency and replicability for reviewers.
**The output screenshots captured during the run-time of this example can be found at the following link: https://github.com/hiweir/AutoRSP-Supplementary-Material/blob/main/AutoRSP-Supplementary-Material-03RSP-generate-example.docx.**

------

## A. Experimental Setup

- **Platform & Environment**: Ubuntu 20.04 LTS, ROS Noetic, Gazebo 11.15.1, Python 3.11
- **Orchestration**: LangGraph (workflow scheduling) + LangSmith (trace visualization)
- **LLM Configuration**: GLM-4.7, *temperature=0* (deterministic), max retries=3
- **Input Dataset**: *turtlebot3_example* (9 Python files, 35 raw robot control functions)

------

## B. Input Configuration

The workflow is initialized via the LangGraph web interface with three core parameters:

1. **Input Dir**: Path to the raw TurtleBot3 codebase

2. **Output File**: Target path for the final RSP library (*turtlebot3_example_01.py*)

3. **Max Iterations**: 6 (upper bound for the evaluation-optimization loop)

   

------

## C. Step-by-Step Workflow Execution

AutoRSP executes four sequential core nodes, with real-time state logging and trace visualization.

### 1. Function Extraction Node

- Parses all 9 Python files in the input directory via AST static analysis.
- Extracts **35 explicit functions** and detects **0 standalone procedural scripts**.
- Outputs structured function metadata (name, code, dependencies, file path).



### 2. Description Generation Node

- Generates semantic descriptions for all extracted functions.
- Resolves homonym conflicts: renames *main* → *createTurtlebotNode*, *getkey* → *checkAndShutdown*.
- Merges 8 duplicate ***init*** functions and deduplicates to **28 unique valid functions**.



### 3. Domain-aware Refactoring Node

- Consolidates 28 functions into a unified ROS-compatible class: *Turtlebot3RobotSkillPrimitives*.
- Injects hard domain constraints: ROS node initialization, TurtleBot3 velocity limits, and Gazebo topic subscriptions (*cmd_vel, odom, scan*).
- Preserves 1:1 functional mapping with raw input code.



### 4. Iterative Evaluation & Optimization Node

- Runs **3 optimization iterations** (within the 6-iteration limit).
- Fixes critical issues incrementally: 5 → 1 → 0 critical defects.
- Validates syntax, ROS compatibility, hardware constraints, and simulation readiness.
- Terminates loop when code is marked *verified*.



------

## D. Final Output

AutoRSP generates *turtlebot3_example_01.py*, a production-ready RSP library featuring:

1. A verified *Turtlebot3RobotSkillPrimitives* class with 28 executable skill primitives.
2. Full ROS/Gazebo compatibility and safety guard implementations.
3. Structured verification reports and iteration logs.
4. Directly deployable code for TurtleBot3 simulation and physical deployment.

------

## E. Key Observations

1. **Full Automation**: Zero manual intervention across the entire code-to-RSP pipeline.
2. **Constraint Compliance**: Output strictly adheres to ROS Noetic standards and TurtleBot3 hardware limits.
3. **Iterative Robustness**: The closed-loop loop eliminates defects missed in one-shot refactoring.
4. **Reproducibility**: Consistent results with the main paper’s experimental metrics.

This end-to-end example validates that AutoRSP reliably transforms unstructured robotic codebases into high-quality, verified RSP libraries as claimed in the manuscript.