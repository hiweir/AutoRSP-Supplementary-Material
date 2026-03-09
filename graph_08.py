import os
import re
import ast
import json
import logging
import time
from typing import Dict, Tuple, Literal, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from pathlib import Path
from dataclasses import dataclass, field, asdict

# LangChain核心组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph状态机
from langgraph.graph import StateGraph
from langgraph.constants import END

# from my_llm import llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ====================== API配置 ======================
ZHIPU_API_KEY = "0660402d927c4d38831835688eeeb0d0.4VCoA33RlilSGtwZ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

QWEN_API_KEY = "sk-459eecb8502547f6bf6c2fdfa8c4ab2f"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEEPSEEK_API_KEY="sk-2589c4bfddab493aa0256ce2be29b093"
DEEPSEEK_BASE_URL="https://api.deepseek.com"

# llm1 = ChatOpenAI(
#     model='deepseek-reasoner',    # 对应DeepSeek-V3.2 最新版本
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     temperature=0,
#     max_retries=3,
#     request_timeout=180,
# )

# llm = ChatOpenAI(
#     model='deepseek-reasoner',
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
#     temperature=0,
#     max_retries=3,
#     request_timeout=180,
# )

llm1 = ChatOpenAI(
    model='glm-4.5-flash',
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0,
    max_retries=3,
    request_timeout=180,
)

llm = ChatOpenAI(
    model='glm-4.5-flash',
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0,
    max_retries=3,
    request_timeout=180,
)


# ====================== 数据结构定义 ======================
@dataclass
class FunctionMetadata:
    name: str
    code: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    file_path: str = ""
    is_standalone: bool = False


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, FunctionMetadata):
            return asdict(obj)
        return super().default(obj)


# 状态State对象定义
class ProcessingState(TypedDict):
    input_dir: str  # 输入目录
    output_file: str  # 输出文件
    extracted_functions: List[FunctionMetadata]  # 提取的函数
    standalone_files: List[Tuple[str, str]]  # 没有类函数的py独立文件
    selected_functions: List[FunctionMetadata]  # 去重之后的函数
    init_functions: List[FunctionMetadata]  # 发现的init函数
    function_descriptions: Dict[str, str]  # 函数的功能描述
    refactored_code: str  # 重构生成的函数代码
    optimized_code: str  # 优化后的函数代码
    verification_report: str  # 生成的验证报告反馈
    final_output: str  # 最终的输出内容（机器人的技能原语库）
    iteration_count: int  # 代码质量验证的迭代次数
    max_iterations: int  # 代码质量验证的最大迭代次数
    needs_more_work: bool  # 判断是否继续代码优化的标志（值为True:表示代码需要继续优化；值为False：表示代码不需要继续优化）
    # feedback: str                                  #重构后函数代码的优化建议反馈
    robots_type: str  # 根据功能重构统一类函数的机器人类型
    verified_or_not: str  # 重构后函数代码的质量验证是否通过
    critical_issues: int  # 验证报告中存在的严重问题数量及问题描述总结
    warning_issues: int  # 验证报告中发现的警告问题及问题描述总结


# 结构化输出模型（用于LLM评估反馈）
# 首先，我们定义统一的结构化验证报告模型（与第一段代码风格保持一致）
class StructuredVerificationReport(BaseModel):
    """使用此工具来结构化你的验证响应"""
    verified_or_not: Literal["verified", "not verified"] = Field(
        description="判断重构后的TurtleBot3机器人的类函数代码质量是否通过验证",
        examples=["verified", "not verified"]
    )
    verification_report: str = Field(
        description="详细的验证报告，包含一致性、语法、ROS兼容性、测试用例等所有方面的检查结果与优化建议",
        example="函数一致性检查: 通过\n语法检查: 发现3处未定义变量\nROS兼容性: 缺少rospy初始化\n测试用例: 为move方法生成测试输入..."
    )
    critical_issues: int = Field(
        description="验证报告中发现的严重问题数量及问题描述总结（如语法错误、ROS兼容性失败）",
        example="重构后的代码存在2个严重问题，分别如下：1.语法错误。\n2.ROS兼容性失败..."
    )
    warning_issues: int = Field(
        description="验证报告中发现的警告问题数量（不影响代码正常运行）及问题描述总结（如函数名称不一致但功能正确）",
        example="重构后的代码存在1个警告问题（不影响代码正常运行），分别如下：1.函数名称不一致但功能正确"
    )


# ====================== 工具函数 ======================
def extract_imports(code: str) -> List[str]:
    """从Python代码中提取所有导入语句"""
    try:
        tree = ast.parse(code)
        imports = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return list(set(imports))
    except SyntaxError as e:
        logger.warning(f"语法错误在提取导入时: {str(e)}")
        return []


def format_function_metadata(functions: List[FunctionMetadata]) -> str:
    """格式化函数元数据为字符串"""
    return "\n\n".join(
        f"# 文件: {f.file_path}\n函数名: {f.name}\n代码:\n```python\n{f.code}\n```"
        for f in functions
    )


def get_valid_path(prompt: str) -> Path:
    """获取有效的输入路径"""

    while True:
        path_input = input(prompt).strip()
        if not path_input:
            print("⛔ 输入不能为空，请重新输入")
            continue

        path = Path(path_input)
        if path.exists() and path.is_dir():
            return path

        print(f"⛔ 路径无效或不是目录: {path_input}")


def get_valid_output_path(prompt: str) -> Path:
    """获取有效的输出文件路径"""
    while True:
        path_input = input(prompt).strip()
        if not path_input:
            return Path("robot_skill_primitives.py")

        path = Path(path_input)

        if path.is_dir():
            print(f"⚠️ 您输入的是目录，将在其中创建 robot_skill_primitives.py")
            return path / "robot_skill_primitives.py"

        if path.suffix.lower() != ".py":
            path = path.with_suffix(".py")
            print(f"⚠️ 自动添加.py扩展名: {path}")

        parent = path.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建目录: {parent}")
            except Exception as e:
                print(f"⛔ 无法创建目录 {parent}: {str(e)}")
                continue

        if not os.access(str(parent), os.W_OK):
            print(f"⛔ 没有写入权限: {parent}")
            continue

        return path


def generate_fallback_class(functions: List[FunctionMetadata]) -> str:
    """生成基本类结构（增强降级方案）"""
    logger.warning("🛠 生成增强降级类结构...")

    all_dependencies = set()
    for func in functions:
        all_dependencies.update(func.dependencies)

    # 针对TurtleBot3 的增强导入
    import_lines = [
        "import rospy",
        "import numpy as np",
        "import tf",
        "from geometry_msgs.msg import Twist, Point",
        "from sensor_msgs.msg import LaserScan",
        "from nav_msgs.msg import Odometry",
        "# 注意: 请根据实际安装情况导入TurtleBot3包，例如:",
        "# from turtlebot3_bringup.msg import RobotState",
        "# from turtlebot3_gazebo.msg import GazeboState"
    ]

    class_code = [
        "# 机器人技能原语库（降级方案）- 专为TurtleBot3 优化",
        "# 注意：这是一个基本类结构，需要根据实际需求完善实现",
        "\n".join(import_lines),
        "",
        "class Turtlebot3RobotSkillPrimitives:",
        "    def __init__(self, node_name='turtlebot3_skill_primitives'):",
        "        # ROS节点初始化 - Gazebo测试必需",
        "        rospy.init_node(node_name, anonymous=True)",
        "        ",
        "        # 创建速度发布器 - 用于控制机器人移动",
        "        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)",
        "        ",
        "        # 订阅激光雷达数据 - 用于避障等任务",
        "        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)",
        "        self.latest_scan = None",
        "        ",
        "        # 订阅里程计数据 - 用于导航",
        "        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)",
        "        self.current_pose = None",
        "        ",
        "        # 初始化Twist消息",
        "        self.twist_msg = Twist()",
        "        ",
        "        # TurtleBot3 安全参数 (单位: 米, 弧度/秒)",
        "        self.SAFE_STOP_DISTANCE = 0.3  # 建议停止距离",
        "        self.MAX_LINEAR_VEL = 0.22     # 最大线速度",
        "        self.MAX_ANGULAR_VEL = 2.84    # 最大角速度",
        "        ",
        "        # 控制循环速率 (单位: Hz)",
        "        self.rate = rospy.Rate(10) # 10Hz",
        "        ",
        "        rospy.loginfo('Turtlebot3RobotSkillPrimitives 初始化完成')",
        "",
        "    def scan_callback(self, msg):",
        "        \"\"\"处理激光雷达数据\"\"\"",
        "        self.latest_scan = msg",
        "",
        "    def odom_callback(self, msg):",
        "        \"\"\"处理里程计数据\"\"\"",
        "        self.current_pose = msg.pose.pose",
        "",
        "    def emergency_stop(self):",
        "        \"\"\"紧急停止机器人\"\"\"",
        "        self.twist_msg.linear.x = 0.0",
        "        self.twist_msg.angular.z = 0.0",
        "        self.cmd_vel_pub.publish(self.twist_msg)",
    ]

    for func in functions:
        class_code.append(f"    def {func.name}(self, *args, **kwargs):")
        class_code.append(f"        \"\"\"{func.description}\"\"\"")
        class_code.append(f"        # 原始代码位置: {func.file_path}")
        if func.is_standalone:
            class_code.append("        # 注意: 此函数由独立代码封装生成")
        class_code.append("        # 降级方案: 请根据实际需求实现此方法")
        class_code.append("        rospy.loginfo(f'调用 {func.name} 方法')")
        class_code.append("        # 示例实现 - 发布停止指令")
        class_code.append("        self.emergency_stop()")
        class_code.append("        return {'status': 'success', 'message': 'Function called (fallback)'}\n")

    return "\n".join(class_code)


def refactor_with_retry(chain, input_data: dict, max_retries: int = 3, timeout: int = 240) -> str:
    """带重试机制的API调用（增强超时处理）"""
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            result = chain.invoke(input_data)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"请求处理时间过长 ({elapsed:.1f}s)，尝试简化请求")
                input_data["functions_list"] = input_data["functions_list"][:500] + "..."
                continue

            return result
        except Exception as e:
            if "502" in str(e) or "Bad Gateway" in str(e):
                wait_time = 2 ** attempt
                logger.warning(f"⏳ API网关错误 (502), 第{attempt + 1}次重试, 等待{wait_time}秒...")
                time.sleep(wait_time)
            elif "timed out" in str(e) or "timeout" in str(e):
                wait_time = 2 ** attempt
                logger.warning(f"⏳ 请求超时, 第{attempt + 1}次重试, 等待{wait_time}秒...")
                time.sleep(wait_time)
            else:
                raise e
    raise ConnectionError(f"API请求失败，已达最大重试次数{max_retries}")


# ====================== 处理节点函数 ======================
def extract_and_process_files(state: ProcessingState) -> ProcessingState:
    """节点1: 提取函数并处理独立文件（合并原提取函数和处理独立文件节点）"""
    logger.info(f"🔍 开始解析目录: {state['input_dir']}")
    extracted_functions = []
    file_count = 0
    standalone_files = []

    # 原code_parser功能
    for py_file in Path(state["input_dir"]).rglob('*.py'):
        file_count += 1
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)
            has_functions = False

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    has_functions = True
                    func_code = ast.unparse(node)
                    extracted_functions.append(FunctionMetadata(
                        name=node.name,
                        code=func_code,
                        dependencies=extract_imports(code),
                        file_path=str(py_file)
                    ))

            if not has_functions and not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree)):
                standalone_files.append((str(py_file), code))
                logger.info(f"  📄 检测到独立代码文件: {py_file}")

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"解析错误 {py_file}: {str(e)}")
            continue

    state["extracted_functions"] = extracted_functions
    state["standalone_files"] = standalone_files
    logger.info(
        f"✅ 提取完成! 在 {file_count} 个文件中发现 {len(extracted_functions)} 个函数和 {len(standalone_files)} 个独立文件")

    # 原process_standalone_files功能
    if not state.get("standalone_files"):
        return state

    logger.info("📦 开始封装独立代码文件...")
    standalone_functions = []

    standalone_prompt = ChatPromptTemplate.from_messages([
        ("system", "您是一个代码重构专家。请将以下独立代码封装为一个函数：\n"
                   "1. 根据代码功能生成合适的函数名（动词+名词形式）\n"
                   "2. 避免与现有函数重名\n"
                   "3. 添加必要的函数参数"
                   "4. 添加函数描述文档\n"
                   "5. 保持原始功能不变\n\n"
                   "输出格式：\n"
                   "函数名: <生成的函数名>\n"
                   "函数代码: ```python\n<封装后的代码>\n```"),
        ("human", "独立代码：\n```python\n{code}\n```\n\n文件路径: {file_path}")
    ])

    chain = standalone_prompt | llm1 | StrOutputParser()

    for file_path, code in state["standalone_files"]:
        try:
            logger.info(f"  封装文件: {Path(file_path).name}")
            result = chain.invoke({"code": code, "file_path": file_path})

            func_name_match = re.search(r"函数名:\s*(\w+)", result)
            code_match = re.search(r"```python\n(.*?)\n```", result, re.DOTALL)

            if func_name_match and code_match:
                func_name = func_name_match.group(1)
                func_code = code_match.group(1)

                standalone_func = FunctionMetadata(
                    name=func_name,
                    code=func_code,
                    dependencies=extract_imports(code),
                    file_path=file_path,
                    is_standalone=True
                )
                standalone_functions.append(standalone_func)
                logger.info(f"    ✓ 生成函数: {func_name}")
            else:
                logger.warning(f"    ⚠️ 无法解析输出: {result[:100]}...")
        except Exception as e:
            logger.error(f"封装独立代码错误 {file_path}: {str(e)}")

    if standalone_functions:
        state["extracted_functions"].extend(standalone_functions)
        logger.info(f"✅ 成功封装 {len(standalone_functions)} 个独立代码文件")

    return state


def generate_function_descriptions(state: ProcessingState) -> ProcessingState:
    """节点2: 生成函数描述并处理重名函数"""
    if not state.get("extracted_functions"):
        logger.warning("⛔ 没有可处理的函数，跳过描述生成")
        return state

    logger.info("🧠 开始生成函数描述并处理重名函数...")
    descriptions = {}
    selected_functions = []
    init_functions = []

    cot_prompt = ChatPromptTemplate.from_messages([
        ("system", "您是一个机器人代码重构专家。请为以下函数生成一个简洁的自然语言功能描述，"
                   "不要包含分析过程或步骤说明，直接描述函数的核心功能和需要传入的参数。"),
        ("human", "函数代码：\n```python\n{function_code}\n\n```\n"
                  "请直接输出功能描述和需要传入的参数，不要包含其他内容。")
    ])

    chain = cot_prompt | llm1 | StrOutputParser()

    compare_prompt = ChatPromptTemplate.from_messages([
        ("system", "您是一个代码分析专家。请比较两个同名函数的功能差异：\n"
                   "1. 如果功能完全相同，输出『相同』\n"
                   "2. 如果功能不同，输出『不同』并生成一个基于功能的新函数名（使用动词+名词形式，不超过25字符）\n"
                   "输出格式：\n"
                   "功能比较: [相同/不同]\n"
                   "新函数名: [新名称] (仅当不同时)"),
        ("human", "函数1描述: {desc1}\n函数2描述: {desc2}\n函数原名: {func_name}")
    ])
    compare_chain = compare_prompt | llm1 | StrOutputParser()

    rename_prompt = ChatPromptTemplate.from_messages([
        ("system", "您是一个代码重构专家。请根据函数功能生成简洁的英文函数名，遵循：\n"
                   "1. 使用动词+名词形式（如move_to_position）\n"
                   "2. 长度不超过25字符\n"
                   "3. 体现核心功能"),
        ("human", "函数功能描述: {description}")
    ])
    rename_chain = rename_prompt | llm1 | StrOutputParser()

    name_to_funcs = {}
    for func in state["extracted_functions"]:
        if func.name == "__init__":
            init_functions.append(func)
            continue

        if func.name not in name_to_funcs:
            name_to_funcs[func.name] = []
        name_to_funcs[func.name].append(func)

    for func_name, func_list in name_to_funcs.items():
        base_func = func_list[0]
        try:
            desc = chain.invoke({"function_code": base_func.code})
            desc = re.sub(r"分析：|步骤：|功能：|描述：|这个函数|该函数", "", desc).strip()
            base_func.description = desc
            selected_functions.append(base_func)
            descriptions[func_name] = desc
        except Exception as e:
            logger.error(f"生成描述错误 {base_func.name}: {str(e)}")
            continue

        if len(func_list) > 1:
            for i in range(1, len(func_list)):
                func = func_list[i]
                try:
                    current_desc = chain.invoke({"function_code": func.code})
                    current_desc = re.sub(r"分析：|步骤：|功能：|描述：|这个函数|该函数", "", current_desc).strip()

                    comparison = compare_chain.invoke({
                        "desc1": desc,
                        "desc2": current_desc,
                        "func_name": func_name
                    })

                    if "功能比较: 相同" in comparison:
                        logger.info(f"  跳过功能相同的重复函数: {func_name} (来自 {func.file_path})")
                        continue
                    elif "新函数名:" in comparison:
                        new_name_match = re.search(r"新函数名:\s*(\w+)", comparison)
                        if new_name_match:
                            new_name = new_name_match.group(1)
                            if new_name in name_to_funcs:
                                new_name = rename_chain.invoke({"description": current_desc})
                            func.name = new_name
                        else:
                            new_name = rename_chain.invoke({"description": current_desc})
                            func.name = new_name
                    else:
                        new_name = rename_chain.invoke({"description": current_desc})
                        func.name = new_name

                    func.description = current_desc
                    selected_functions.append(func)
                    descriptions[new_name] = current_desc
                    logger.info(f"  ✓ 重命名函数: {func_name} -> {func.name} (来自 {func.file_path})")

                except Exception as e:
                    logger.error(f"处理重名函数错误 {func_name}: {str(e)}")
                    func.name = f"{func_name}_{i + 1}"
                    selected_functions.append(func)
                    descriptions[func.name] = current_desc
                    logger.info(f"  ⚠️ 异常处理重命名: {func_name} -> {func.name}")

    if init_functions:
        logger.info(f"  发现 {len(init_functions)} 个__init__函数，准备合并")
        init_descriptions = []
        for init_func in init_functions:
            try:
                desc = chain.invoke({"function_code": init_func.code})
                desc = re.sub(r"分析：|步骤：|功能：|描述：|这个函数|该函数", "", desc).strip()
                init_descriptions.append(desc)
            except Exception as e:
                logger.error(f"生成__init__描述错误: {str(e)}")
                init_descriptions.append("初始化函数")

        merged_init = FunctionMetadata(
            name="__init__",
            code="",
            description="整合初始化逻辑，包括: " + "; ".join(init_descriptions),
            dependencies=[],
            file_path="合并多个文件"
        )
        selected_functions.append(merged_init)
        descriptions["__init__"] = merged_init.description

    state["function_descriptions"] = descriptions
    state["selected_functions"] = selected_functions
    state["init_functions"] = init_functions
    logger.info(f"✅ 降重后唯一函数数量: {len(selected_functions)}")
    return state


def refactor_code(state: ProcessingState) -> ProcessingState:
    """节点3: 重构代码为统一类"""
    if not state.get("selected_functions") or len(state["selected_functions"]) == 0:
        logger.warning("⛔ 没有可处理的函数，跳过重构")
        state["refactored_code"] = "# 重构失败: 无有效函数\n"
        return state

    logger.info("🔄 开始重构为统一类...")

    all_dependencies = set()
    for func in state["selected_functions"]:
        all_dependencies.update(func.dependencies)
    for func in state.get("init_functions", []):
        all_dependencies.update(func.dependencies)

    # 增强的重构提示词，强调TurtleBot3 和Gazebo测试，特别是位姿记录
    refactor_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "您是一个专业的ROS机器人代码架构师，专注于TurtleBot3机器人的精确代码生成。请严格按照以下要求重构函数：\n\n"

         "## 核心重构原则\n"
         "1. **精确性优先**：每个函数必须基于原始代码逻辑，不添加无关功能\n"
         "2. **TurtleBot3硬件约束**：线速度≤0.22m/s，角速度≤2.84rad/s\n"
         "3. **ROS Noetic规范**：使用标准ROS 1消息类型和编程模式\n\n"

         "## 具体实现要求\n"
         "### 类结构设计\n"
         "- 类名：`Turtlebot3RobotSkillPrimitives`\n"
         "- 必须包含的成员变量：\n"
         "  - `cmd_vel_pub`: 发布到/cmd_vel的Publisher\n"
         "  - `current_pose`: 从/odom更新的当前位置\n"
         "  - `latest_scan`: 从/scan更新的激光数据\n"
         "  - `MAX_LINEAR_VEL = 0.22`, `MAX_ANGULAR_VEL = 2.84`\n\n"

         "### 初始化方法(__init__)\n"
         "必须包含：\n"
         "- `rospy.init_node('turtlebot3_skill_primitives')`\n"
         "- 发布器：`self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)`\n"
         "- 订阅器：\n"
         "  - `/odom → self.odom_callback`（更新self.current_pose）\n"
         "  - `/scan → self.scan_callback`（更新self.latest_scan）\n"
         "- 安全参数初始化\n\n"

         "### 函数重构规则\n"
         "1. **一对一映射**：每个原始函数对应一个类方法，保持相同功能\n"
         "2. **参数传递**：原函数参数直接转换为方法参数\n"
         "3. **返回值处理**：原函数返回值类型必须保持一致\n"
         "4. **异常处理**：每个方法必须包含try-catch和rospy日志\n"
         "5. **速度限制**：所有运动控制方法必须包含速度边界检查\n\n"

         "### Gazebo测试要求\n"
         "- 所有传感器话题必须正确订阅和回调\n"
         "- 控制指令必须通过/cmd_vel发布Twist消息\n"
         "- 必须包含位姿更新逻辑：\n"
         "  ```python\n"
         "  def odom_callback(self, msg):\n"
         "      self.current_pose = msg.pose.pose  # 更新当前位置\n"
         "  ```\n\n"

         "## 输出格式\n"
         "只输出完整的Python类代码，包含：\n"
         "1. 精确的导入语句（仅使用实际需要的ROS包）\n"
         "2. 完整的类定义\n"
         "3. 基于原始函数列表的精确方法实现\n"
         "4. 必要的注释说明\n\n"

         "## 禁止事项\n"
         "- 禁止添加原始函数列表之外的功能\n"
         "- 禁止修改原始函数的输入输出接口\n"
         "- 禁止使用全局变量或外部依赖\n"
         "- 禁止忽略TurtleBot3的物理约束\n\n"

         "现在开始基于以下输入进行精确重构："),

        ("human", "函数列表（功能参考）：\n{functions_list}\n\n"
                  "__init__函数列表（需合并）：\n{init_functions_list}\n\n"
                  "依赖库：\n{dependencies}\n\n"
                  "请输出完整类代码，包含所有导入语句。")
    ])

    chain = refactor_prompt | llm | StrOutputParser()

    try:
        functions_list = format_function_metadata(
            [f for f in state["selected_functions"] if f.name != "__init__"]
        )
        init_functions_list = format_function_metadata(
            state.get("init_functions", []) +
            [f for f in state["selected_functions"] if f.name == "__init__"]
        )

        refactored_code = refactor_with_retry(
            chain,
            {
                "functions_list": functions_list,
                "init_functions_list": init_functions_list,
                "dependencies": ", ".join(all_dependencies)
            },
            max_retries=5,
            timeout=300
        )

        state["refactored_code"] = re.sub(r"```python|```", "", refactored_code).strip()
        logger.info("✅ 重构成功! 生成统一类代码")
    except Exception as e:
        logger.error(f"重构失败: {str(e)}")
        state["refactored_code"] = generate_fallback_class(state["selected_functions"])
        if state.get("init_functions"):
            init_comment = "\n# 注意: 需要手动合并以下__init__函数:\n"
            for init_func in state["init_functions"]:
                init_comment += f"# 文件: {init_func.file_path}\n"
            state["refactored_code"] = init_comment + state["refactored_code"]

    return state


def optimize_and_evaluate(state: ProcessingState) -> ProcessingState:
    """节点4: 评估优化器（迭代循环体）"""
    # 检查是否有有效重构代码
    if not state.get("refactored_code") or "重构失败" in state["refactored_code"]:
        logger.warning("⛔ 无有效重构代码，跳过验证")
        # 使用结构化报告，即使失败也返回统一格式
        state["verification_report"] = "无有效重构代码"
        state["verified_or_not"] = "not verified"
        state["critical_issues"] = 1  # 无代码本身就是一个严重问题
        state["warning_issues"] = 0
        return state

    # 初始化或获取迭代计数器
    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count
    max_iterations = state.get("max_iterations", 5)
    logger.info(f"🔄 当前迭代次数: {iteration_count}/{max_iterations}")

    # 检查是否达到最大迭代次数
    if iteration_count >= max_iterations:
        logger.warning("⏰ 已达到最大迭代次数，停止优化循环")
        state["needs_more_work"] = False
        return state

    # 根据是否存在验证报告，决定是执行优化还是验证
    if state.get("verification_report"):
        logger.info("🛠️ 检测到验证报告，进入优化模式...")
        # 优化代码
        optimization_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "您是一个精确的ROS机器人代码优化专家，专注于TurtleBot3机器人的代码质量提升。请基于验证报告进行针对性修复：\n\n"

             "## 优化原则\n"
             "1. **精准修复**：只修复验证报告中指出的具体问题，不改变原始功能逻辑\n"
             "2. **最小改动**：保持代码结构稳定，避免不必要的重构\n"
             "3. **TurtleBot3约束保持**：确保所有优化不违反硬件限制（线速度0.22m/s，角速度2.84rad/s）\n\n"

             "## 具体修复指南\n"
             "### 语法错误修复\n"
             "- 未定义变量：检查变量作用域，在__init__或方法内正确定义\n"
             "- 语法错误：逐行检查Python语法，确保缩进、括号匹配\n"
             "- 导入缺失：仅添加验证报告指出的必要ROS包，避免过度导入\n\n"

             "### ROS兼容性修复\n"
             "- 节点初始化：确保`rospy.init_node`在__init__中正确调用\n"
             "- 话题配置：验证/cmd_vel、/odom、/scan话题的发布订阅关系\n"
             "- 消息类型：检查Twist、LaserScan、Odometry等消息的正确使用\n\n"

             "### TurtleBot3特定修复\n"
             "- 速度限制：所有运动控制必须包含边界检查\n"
             "  ```python\n"
             "  linear_vel = max(min(linear_vel, self.MAX_LINEAR_VEL), -self.MAX_LINEAR_VEL)\n"
             "  angular_vel = max(min(angular_vel, self.MAX_ANGULAR_VEL), -self.MAX_ANGULAR_VEL)\n"
             "  ```\n"
             "- 位姿更新：确保odom_callback正确更新self.current_pose\n"
             "- 安全停止：保留emergency_stop方法并正确实现\n\n"

             "### 依赖包清理\n"
             "- 移除未使用的导入语句\n"
             "- 合并重复导入（如import rospy和from rospy import Time）\n"
             "- 仅保留实际被调用的ROS包\n\n"

             "## 优化验证要求\n"
             "优化后必须通过以下检查：\n"
             "1. 语法检查：无语法错误，可通过Python AST解析\n"
             "2. 功能检查：所有原始函数逻辑保持不变\n"
             "3. ROS检查：符合ROS Noetic标准，话题配置正确\n"
             "4. 约束检查：满足TurtleBot3硬件限制\n\n"

             "## 输出格式\n"
             "输出优化后的完整类代码，并在代码末尾添加优化总结：\n"
             "'''\n"
             "优化总结：\n"
             "- 修复问题：[列出具体修复的问题]\n"
             "- 保持功能：[确认原始功能完整性]\n"
             "- 验证状态：[优化后是否通过验证]\n"
             "'''\n\n"

             "现在基于以下验证报告进行精确优化："),

            ("human", "待优化代码：\n```python\n{code}\n```"
                      "验证报告：\n{verification_report}\n\n")
        ])

        chain = optimization_prompt | llm | StrOutputParser()

        try:
            optimized_code = refactor_with_retry(
                chain,
                {
                    "code": state["refactored_code"],
                    "verification_report": state["verification_report"]
                },
                max_retries=3
            )

            state["optimized_code"] = re.sub(r"```python|```", "", optimized_code).strip()
            logger.info("✅ 优化成功! 生成优化类代码")

            # 将优化后的代码设置为新的重构代码，为下一次迭代准备
            state["refactored_code"] = state["optimized_code"]

        except Exception as e:
            logger.error(f"优化失败: {str(e)}")
            state["optimized_code"] = state["refactored_code"]

    # 执行验证
    logger.info("🔬 开始代码验证...")

    # 构建统一的验证提示词
    unified_verification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "您是一个严格的机器人代码质量验证专家，专注于TurtleBot3机器人的精确验证。请按照以下结构化标准进行逐项验证：\n\n"

         "## 验证标准（逐项检查，必须提供具体证据）\n"
         "### 1. 函数一致性验证（核心检查项）\n"
         "- 检查重构后的类是否包含所有原始函数：{functions_list}\n"
         "- 验证每个函数的参数列表是否与原始函数一致\n"
         "- 检查返回值类型是否匹配原始函数\n"
         "- 确认功能逻辑是否保持完整\n\n"

         "### 2. 语法与静态检查（必须零容忍）\n"
         "- 使用Python AST解析验证语法正确性\n"
         "- 检查所有变量是否正确定义和初始化\n"
         "- 验证导入语句是否完整且必要\n"
         "- 确认无死代码或未使用变量\n\n"

         "### 3. ROS Noetic兼容性验证（关键检查项）\n"
         "- 必须包含：`rospy.init_node('turtlebot3_skill_primitives')`\n"
         "- 必须包含：`self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)`\n"
         "- 必须包含：`self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)`\n"
         "- 必须包含：`self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)`\n"
         "- 验证所有ROS消息类型正确导入和使用\n\n"

         "### 4. TurtleBot3硬件约束验证（非妥协项）\n"
         "- 检查是否定义：`self.MAX_LINEAR_VEL = 0.22`\n"
         "- 检查是否定义：`self.MAX_ANGULAR_VEL = 2.84`\n"
         "- 验证所有运动控制方法是否包含速度边界检查\n"
         "- 确认紧急停止方法`emergency_stop`正确实现\n\n"

         "### 5. Gazebo仿真准备验证（关键功能）\n"
         "- 验证`odom_callback`方法是否正确更新`self.current_pose`\n"
         "- 检查`scan_callback`方法是否正确处理激光数据\n"
         "- 确认控制循环速率设置合理（通常10Hz）\n"
         "- 验证话题发布/订阅关系正确\n\n"

         "### 6. 代码质量验证（可维护性）\n"
         "- 检查异常处理是否完整（try-catch + rospy日志）\n"
         "- 验证注释和文档字符串是否清晰\n"
         "- 确认代码结构合理，无冗余逻辑\n\n"

         "## 验证报告格式要求\n"
         "对于每个检查项，必须提供：\n"
         "- 检查结果：[通过/失败]\n"
         "- 具体证据：代码行号或具体实现\n"
         "- 问题描述：如果失败，详细说明问题\n\n"

         "## 验证结论标准\n"
         "只有满足以下所有条件才能标记为'verified'：\n"
         "1. 函数一致性验证全部通过\n"
         "2. 语法检查零错误\n"
         "3. ROS兼容性关键项全部通过\n"
         "4. TurtleBot3硬件约束全部满足\n"
         "5. Gazebo仿真准备核心功能正常\n\n"

         "现在开始严格验证以下代码，请根据上述检查点输出一份全面的验证报告。"),

        ("human", "待验证代码：\n```python\n{code}\n```\n\n"
                  "原始函数列表: {functions_list}")
    ])

    # 使用LLM绑定结构化报告工具
    chain = unified_verification_prompt | llm.bind_tools([StructuredVerificationReport])

    try:
        # 准备输入
        functions_list = [f.name for f in state.get("selected_functions", [])]
        current_code = state.get("optimized_code", state["refactored_code"])  # 优先使用优化后的代码进行验证
        input_content = {
            "code": current_code,
            "functions_list": functions_list
        }

        # 调用LLM获取结构化响应
        evaluation = chain.invoke(input_content)
        # 提取工具调用的参数
        if evaluation.tool_calls:
            report_args = evaluation.tool_calls[-1]['args']
            unified_grade = report_args['verified_or_not']
            unified_report = report_args['verification_report']
            critical_issues = report_args['critical_issues']
            warning_issues = report_args['warning_issues']
        else:
            # 如果LLM没有调用工具，降级处理
            unified_grade = "not verified"
            unified_report = "LLM未返回结构化验证报告"
            critical_issues = 1
            warning_issues = 0

        # 存储统一验证结果
        state["verification_report"] = unified_report
        state["verified_or_not"] = unified_grade
        state["critical_issues"] = critical_issues
        state["warning_issues"] = warning_issues

        logger.info("✅ 验证完成!")

    except Exception as e:
        logger.error(f"验证失败: {str(e)}")
        state["verification_report"] = f"验证失败: {str(e)}"
        state["verified_or_not"] = "not verified"
        state["critical_issues"] = 1
        state["warning_issues"] = 0
        return state

    # 分析验证报告并决定是否需要继续迭代
    logger.info("📊 分析验证报告，决定是否需要继续优化...")

    # 基于结构化数据决策
    critical_issues = state.get("critical_issues", 0)
    warning_issues = state.get("warning_issues", 0)
    verified_or_not = state.get("verified_or_not", "not verified")

    # 决策逻辑：只要有严重问题或验证等级为"not verified"就需要继续工作
    if critical_issues > 0 or verified_or_not == "not verified":
        logger.info(f"🔧 发现 {critical_issues} 个严重问题，需要继续优化")
        state["needs_more_work"] = True
    else:
        logger.info("✅ 代码质量良好，无需进一步优化")
        state["needs_more_work"] = False

    return state


def safe_generate_output(state: ProcessingState) -> ProcessingState:
    """安全版本：生成最终输出"""
    logger.info("📝 生成最终技能原语库...")

    output_path = Path(state["output_file"])

    if output_path.is_dir():
        logger.warning(f"输出路径是目录，转换为文件: {output_path}/robot_skill_primitives.py")
        output_path = output_path / "robot_skill_primitives.py"

    parent_dir = output_path.parent
    if not parent_dir.exists():
        logger.info(f"创建目录: {parent_dir}")
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"无法创建目录 {parent_dir}: {str(e)}")
            output_path = Path.cwd() / "robot_skill_primitives.py"
            logger.info(f"使用回退路径: {output_path}")

    prompt_lines = ['user_prompt = """这里有一些您可以用来指挥TurtleBot3 机器人的函数。']

    for func in state.get("selected_functions", []):
        description = state["function_descriptions"].get(func.name, "")
        if not description:
            description = f"{func.name}函数的功能描述"

        prompt_lines.append(f"Turtlebot3RobotSkillPrimitives.{func.name}() - {description}")

    prompt_lines.append('"""')

    output_code = f"# TurtleBot3  机器人技能原语库\n# 自动生成于 {output_path}\n# 迭代次数: {state.get('iteration_count', 1)}\n# 设计用于ROS Noetic和Gazebo仿真环境\n\n"
    output_code += "\n".join(prompt_lines) + "\n\n"

    if state.get("optimized_code"):
        output_code += state["optimized_code"]
    else:
        output_code += state.get("refactored_code", "# 无重构代码")

    output_code += f"\n\n'''\n验证报告:\n{state.get('verification_report', '无验证报告')}\n'''"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_code)
        logger.info(f"🎉 生成成功! 输出文件: {output_path}")
    except PermissionError as e:
        logger.error(f"⛔ 写入权限错误: {str(e)}")
        home_dir = Path.home()
        fallback_path = home_dir / "turtlebot3_skill_primitives.py"
        logger.info(f"尝试在用户主目录创建: {fallback_path}")
        try:
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write(output_code)
            logger.info(f"✅ 文件成功创建于: {fallback_path}")
            state["output_file"] = str(fallback_path)
        except Exception as e:
            logger.error(f"⛔ 无法在任何位置创建文件: {str(e)}")
            state["final_output"] = output_code
            logger.info("已保存输出到状态，但未写入文件")
            return state

    state["final_output"] = output_code
    state["output_file"] = str(output_path)
    return state


# ====================== LangGraph工作流 ======================
# 定义工作流workflow 相当于 builder
workflow = StateGraph(ProcessingState)
# 工作流添加节点
workflow.add_node("extract_and_process", extract_and_process_files)
workflow.add_node("generate_descriptions", generate_function_descriptions)
workflow.add_node("refactor_code", refactor_code)
workflow.add_node("optimize_and_evaluate", optimize_and_evaluate)
workflow.add_node("generate_output", safe_generate_output)

# 工作流添加边
# 设置工作流入口
workflow.set_entry_point("extract_and_process")

# 主处理流程
workflow.add_edge("extract_and_process", "generate_descriptions")
workflow.add_edge("generate_descriptions", "refactor_code")
workflow.add_edge("refactor_code", "optimize_and_evaluate")


# 修正路由函数
def route_func(state: ProcessingState) -> str:
    """动态路由决策函数"""
    # 如果验证通过（verified）或者不需要更多工作（即needs_more_work为False），则结束（去generate_output）
    if state.get("verified_or_not") == "verified" or not state.get("needs_more_work", False):
        return 'accepted'
    else:
        return 'rejected'


# 修正条件边映射
workflow.add_conditional_edges(
    'optimize_and_evaluate',
    route_func,
    {
        "accepted": "generate_output",  # 合格则结束
        "rejected": "optimize_and_evaluate"  # 不合格则循环 验证反馈优化
    }
)

# 工作流的结束
workflow.add_edge("generate_output", END)

# 工作流进行编译，得到一个graph图
graph = workflow.compile()
