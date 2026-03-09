# Supplementary Material for AutoRSP: A LLM-Driven Framework for Automated Robot Skill Primitive Generation

*Submitted to IEEE Transactions on Software Engineering*

## S1. Introduction

This supplementary material addresses the technical detail gap of prompt engineering in the main manuscript, which is critical for the reproducibility of the AutoRSP framework. We first formalize the core design principles of our Chain-of-Thought (CoT) prompt system and specialized role prompting, then provide the full, reproducible prompt templates for all key nodes in the AutoRSP workflow. All prompts are aligned with the LangGraph-based pipeline in the main manuscript, and have been validated to produce the experimental results reported in the paper.

------

## S2. Core Design Principles of AutoRSP Prompt Engineering

To ensure the reliability, domain compliance, and reproducibility of LLM outputs in the automated RSP generation pipeline, our prompt system follows 5 evidence-based design principles aligned with software engineering and robotics domain requirements:

1. **Role Specialization & Boundary Isolation**: Each prompt strictly defines a single, atomic expert role corresponding to one node in the LangGraph workflow, eliminating cross-role responsibility overlap to reduce LLM hallucination.
2. **Domain Constraint Hardening**: Non-negotiable robotics domain rules (e.g., ROS Noetic specifications, TurtleBot3 hardware limits, safety constraints) are explicitly embedded as mandatory requirements, rather than optional suggestions, to ensure functional executability of generated code.
3. **Structured CoT Reasoning**: All prompts enforce step-by-step reasoning workflows, decomposing complex robotic code processing tasks into verifiable sub-steps to enhance output transparency and predictability.
4. **Output Format Standardization**: Strict, machine-parsable output formats are predefined for all prompts, enabling seamless state passing between LangGraph nodes without manual intervention.
5. **Zero-Shot Reproducibility**: All prompts are designed for zero-shot inference without task-specific fine-tuning, with a fixed LLM temperature of 0 to eliminate randomness and ensure consistent, reproducible results across runs.

------

## S3. Specialized Expert Roles & Full CoT Prompt Templates

Below are the complete prompt templates for the 5 core expert roles in AutoRSP, corresponding to the 4 key processing nodes in the main manuscript. All prompts are implemented via `ChatPromptTemplate` in LangChain, as shown in the open-source implementation of AutoRSP.

### S3.1 Code Refactoring Expert (Function Extraction Node)

**Role Responsibility**: This role complements AST-based explicit function extraction by encapsulating standalone procedural robotic scripts (without formal function definitions) into standardized, reusable functions. It ensures the integrity of original robotic control logic while enforcing consistent naming and documentation conventions.

```
standalone_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional code refactoring expert. Please encapsulate the following standalone robotic code into a standardized function, following these rules strictly:\n"
               "1. Generate a descriptive function name in verb-noun format based on the code's core functionality\n"
               "2. Avoid name duplication with existing functions\n"
               "3. Add necessary function parameters and a complete docstring\n"
               "4. Preserve the original functional logic without any modification\n\n"
               "Output Format:\n"
               "Function Name: <generated function name>\n"
               "Function Code: ```python\n<encapsulated function code>\n```"),
    ("human", "Standalone Code:\n```python\n{code}\n```\n\nSource File Path: {file_path}")
])
```

### S3.2 Code Analysis Expert (Description Generation Node)

**Role Responsibility**: This role performs two core tasks: (1) generating precise semantic descriptions for extracted functions via CoT reasoning; (2) resolving homonym conflicts across codebases via semantic comparison, which is the core of the homonym resolution module validated in the ablation study of the main manuscript.

#### S3.2.1 Function Semantic Description Generation (CoT Prompt)

```
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional robotic code analysis expert. Generate a concise, accurate natural language description for the following function. Directly state the core functionality and required input parameters of the function, without any analysis process or step-by-step explanation."),
    ("human", "Function Code:\n```python\n{function_code}\n\n```\n"
              "Output the function description and required parameters directly, with no additional content.")
])
```

#### S3.2.2 Homonym Function Semantic Comparison Prompt

```
compare_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code semantic analysis expert. Compare the functional differences between two functions with the same name, following these rules:\n"
               "1. If the functions are functionally identical, output only 『Identical』\n"
               "2. If the functions are functionally different, output 『Different』 and generate a new function name in verb-noun format (max 25 characters) based on the functional difference\n\n"
               "Output Format:\n"
               "Functional Comparison: [Identical/Different]\n"
               "New Function Name: [new name] (only if different)"),
    ("human", "Function 1 Description: {desc1}\nFunction 2 Description: {desc2}\nOriginal Function Name: {func_name}")
])
```

#### S3.2.3 Function Renaming Prompt

```
rename_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code refactoring expert. Generate a concise English function name based on the function's functionality, following these rules strictly:\n"
               "1. Use verb-noun format (e.g., move_to_position)\n"
               "2. Maximum length of 25 characters\n"
               "3. Clearly reflect the core functionality of the function"),
    ("human", "Function Description: {description}")
])
```

### S3.3 Robotics Refactoring Expert (Domain-aware Refactoring Node)

**Role Responsibility**: This is the core role of the domain constraint module (validated as the most impactful component in the ablation study). It consolidates extracted functions into a unified, ROS-compatible Python class, with hard-coded robotic domain constraints to ensure syntactic correctness and functional executability in real robotic simulation and physical environments.

```
refactor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional ROS robotic code architect, specializing in precise code generation for TurtleBot3 robots. Refactor the provided functions into a unified Python class, following these mandatory rules strictly:\n\n"
     "## Core Refactoring Principles\n"
     "1. **Precision First**: Each class method must be strictly mapped to the original function, with no additional functionality beyond the original code logic\n"
     "2. **TurtleBot3 Hardware Constraints**: Linear velocity ≤ 0.22 m/s, angular velocity ≤ 2.84 rad/s, enforced in all motion control methods\n"
     "3. **ROS Noetic Compliance**: Use standard ROS 1 message types and programming patterns\n\n"
     "## Mandatory Implementation Requirements\n"
     "### Class Structure\n"
     "- Class Name: `Turtlebot3RobotSkillPrimitives`\n"
     "- Mandatory member variables:\n"
     "  - `cmd_vel_pub`: Publisher for `/cmd_vel` topic\n"
     "  - `current_pose`: Current robot pose updated from `/odom` topic\n"
     "  - `latest_scan`: Latest LiDAR data from `/scan` topic\n"
     "  - `MAX_LINEAR_VEL = 0.22`, `MAX_ANGULAR_VEL = 2.84`\n\n"
     "### Initialization Method (__init__)\n"
     "Must include:\n"
     "- `rospy.init_node('turtlebot3_skill_primitives')`\n"
     "- Publisher: `self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)`\n"
     "- Subscribers:\n"
     "  - `/odom → self.odom_callback` (updates `self.current_pose`)\n"
     "  - `/scan → self.scan_callback` (updates `self.latest_scan`)\n"
     "- Safety parameter initialization\n\n"
     "### Method Refactoring Rules\n"
     "1. 1-to-1 mapping: Each original function corresponds to one class method, preserving original functionality\n"
     "2. Parameter consistency: Original function parameters are directly mapped to method parameters\n"
     "3. Return type consistency: Original return value types are strictly preserved\n"
     "4. Exception handling: Each method must include try-catch blocks and rospy logging\n"
     "5. Velocity limiting: All motion control methods must include velocity boundary checks\n\n"
     "### Gazebo Simulation Compatibility\n"
     "- All sensor topics must have correct subscription and callback implementations\n"
     "- Control commands must be published via `/cmd_vel` as Twist messages\n"
     "- Mandatory pose update logic:\n"
     "  ```python\n"
     "  def odom_callback(self, msg):\n"
     "      self.current_pose = msg.pose.pose\n"
     "  ```\n\n"
     "## Output Format\n"
     "Output only the complete Python class code, including:\n"
     "1. Precise import statements (only ROS packages actually used in the code)\n"
     "2. Complete class definition\n"
     "3. Accurate method implementations based on the provided function list\n"
     "4. Necessary comments and docstrings\n\n"
     "## Prohibited Actions\n"
     "- Do not add functionality beyond the provided function list\n"
     "- Do not modify the input/output interface of original functions\n"
     "- Do not use global variables or external dependencies\n"
     "- Do not violate TurtleBot3 hardware constraints\n\n"
     "Perform the refactoring based on the inputs below:"),
    ("human", "Function List (for functionality reference):\n{functions_list}\n\n"
              "__init__ Function List (to be merged):\n{init_functions_list}\n\n"
              "Dependency Libraries:\n{dependencies}\n\n"
              "Output the complete class code with all import statements.")
])
```

### S3.4 Verification Expert (Evaluation & Optimization Node)

**Role Responsibility**: This role performs structured, multi-dimensional verification of the refactored RSP class, generating a standardized verification report to guide iterative optimization. It aligns with the code quality and downstream utility metrics defined in the main manuscript.

```
unified_verification_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a rigorous robotic code quality verification expert, specializing in TurtleBot3 robot code validation. Perform a line-by-line verification of the provided code against the following structured standards, with concrete evidence for each check item:\n\n"
     "## Verification Standards (item-by-item check, mandatory concrete evidence)\n"
     "### 1. Function Consistency Check (Core Item)\n"
     "- Verify that the refactored class includes all original functions: {functions_list}\n"
     "- Validate that the parameter list of each method matches the original function\n"
     "- Check that the return value type matches the original function\n"
     "- Confirm that the functional logic is fully preserved\n\n"
     "### 2. Syntax & Static Check (Zero Tolerance)\n"
     "- Verify syntax correctness via Python AST parsing\n"
     "- Check that all variables are correctly defined and initialized\n"
     "- Validate that import statements are complete and necessary\n"
     "- Confirm no dead code or unused variables\n\n"
     "### 3. ROS Noetic Compatibility Check (Critical Item)\n"
     "- Must include: `rospy.init_node('turtlebot3_skill_primitives')`\n"
     "- Must include: `self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)`\n"
     "- Must include: `self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)`\n"
     "- Must include: `self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)`\n"
     "- Validate correct import and usage of all ROS message types\n\n"
     "### 4. TurtleBot3 Hardware Constraint Check (Non-Compromisable Item)\n"
     "- Check for definition: `self.MAX_LINEAR_VEL = 0.22`\n"
     "- Check for definition: `self.MAX_ANGULAR_VEL = 2.84`\n"
     "- Verify that all motion control methods include velocity boundary checks\n"
     "- Confirm correct implementation of the `emergency_stop` method\n\n"
     "### 5. Gazebo Simulation Readiness Check (Critical Functionality)\n"
     "- Verify that `odom_callback` correctly updates `self.current_pose`\n"
     "- Check that `scan_callback` correctly processes LiDAR data\n"
     "- Confirm a reasonable control loop rate (typically 10Hz)\n"
     "- Validate correct topic publish/subscribe relationships\n\n"
     "### 6. Code Quality Check (Maintainability)\n"
     "- Check complete exception handling (try-catch + rospy logging)\n"
     "- Validate clear comments and docstrings\n"
     "- Confirm rational code structure with no redundant logic\n\n"
     "## Verification Report Format\n"
     "For each check item, provide:\n"
     "- Check Result: [Pass/Fail]\n"
     "- Concrete Evidence: Line number or specific code implementation\n"
     "- Issue Description: Detailed issue explanation if failed\n\n"
     "## Verification Pass Criteria\n"
     "Only mark as 'verified' if ALL of the following conditions are met:\n"
     "1. All function consistency checks pass\n"
     "2. Zero syntax errors\n"
     "3. All critical ROS compatibility checks pass\n"
     "4. All TurtleBot3 hardware constraints are satisfied\n"
     "5. All core Gazebo simulation readiness functions are normal\n\n"
     "Perform the verification and output a comprehensive report based on the standards above."),
    ("human", "Code to Verify:\n```python\n{code}\n```\n\n"
              "Original Function List: {functions_list}")
])
```

### S3.5 Code Optimization Expert (Evaluation & Optimization Node)

**Role Responsibility**: This role performs targeted, minimal code fixes based on the verification report from the Verification Expert, preserving original functionality while resolving identified defects. It enables the closed-loop iterative optimization validated in the main manuscript.

```
optimization_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precision ROS robotic code optimization expert, specializing in code quality improvement for TurtleBot3 robots. Perform targeted fixes based on the verification report, following these rules strictly:\n\n"
     "## Optimization Principles\n"
     "1. **Precision Fixing**: Only fix the specific issues identified in the verification report, with no changes to the original functional logic\n"
     "2. **Minimal Modification**: Maintain stable code structure, avoid unnecessary refactoring\n"
     "3. **Constraint Preservation**: Ensure all optimizations do not violate TurtleBot3 hardware constraints (linear velocity 0.22m/s, angular velocity 2.84rad/s)\n\n"
     "## Fix Guidelines\n"
     "### Syntax Error Fixes\n"
     "- Undefined variables: Check variable scope, define correctly in __init__ or within the method\n"
     "- Syntax errors: Line-by-line Python syntax check, ensure correct indentation and bracket matching\n"
     "- Missing imports: Only add necessary ROS packages identified in the verification report, avoid excessive imports\n\n"
     "### ROS Compatibility Fixes\n"
     "- Node initialization: Ensure `rospy.init_node` is correctly called in __init__\n"
     "- Topic configuration: Verify publish/subscribe relationships for `/cmd_vel`, `/odom`, `/scan` topics\n"
     "- Message types: Check correct usage of Twist, LaserScan, Odometry, and other ROS messages\n\n"
     "### TurtleBot3-Specific Fixes\n"
     "- Velocity limiting: All motion control must include boundary checks:\n"
     "  ```python\n"
     "  linear_vel = max(min(linear_vel, self.MAX_LINEAR_VEL), -self.MAX_LINEAR_VEL)\n"
     "  angular_vel = max(min(angular_vel, self.MAX_ANGULAR_VEL), -self.MAX_ANGULAR_VEL)\n"
     "  ```\n"
     "- Pose update: Ensure `odom_callback` correctly updates `self.current_pose`\n"
     "- Emergency stop: Preserve and correctly implement the `emergency_stop` method\n\n"
     "### Dependency Cleanup\n"
     "- Remove unused import statements\n"
     "- Merge duplicate imports\n"
     "- Only retain ROS packages that are actually called\n\n"
     "## Post-Optimization Validation Requirements\n"
     "After optimization, the code must pass:\n"
     "1. Syntax check: Zero syntax errors, parsable via Python AST\n"
     "2. Function check: All original function logic remains unchanged\n"
     "3. ROS check: Compliant with ROS Noetic standards, correct topic configuration\n"
     "4. Constraint check: Meets TurtleBot3 hardware limits\n\n"
     "## Output Format\n"
     "Output the complete optimized class code, with an optimization summary at the end of the code:\n"
     "'''\n"
     "Optimization Summary:\n"
     "- Fixed Issues: [list of specific issues fixed]\n"
     "- Preserved Functionality: [confirmation of original functional integrity]\n"
     "- Validation Status: [whether the optimized code passes verification]\n"
     "'''\n\n"
     "Perform the precision optimization based on the verification report below:"),
    ("human", "Code to Optimize:\n```python\n{code}\n```"
              "Verification Report:\n{verification_report}\n\n")
])
```

------

## S4. Workflow Integration & Reproducibility Notes

### S4.1 Prompt-Workflow Integration

All prompts above are bound to the corresponding nodes in the LangGraph workflow of AutoRSP (full implementation in the open-source code repository). Each prompt is executed via a zero-shot call to the GLM-4.7 LLM, with a fixed `temperature=0` to eliminate randomness, `max_retries=3` for API stability, and a structured output parser to ensure seamless state passing between nodes.

### S4.2 Reproducibility Guarantee

The prompt templates provided in this supplementary material are identical to those used in the experiments reported in the main manuscript. Using these prompts with the experimental setup (GLM-4.7, Ubuntu 20.04 LTS, ROS Noetic, Gazebo 11) defined in the main manuscript will fully reproduce the reported code quality and downstream utility results.