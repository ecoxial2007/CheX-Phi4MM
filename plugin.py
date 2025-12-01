# wo xiazaide 
import asyncio
import re
from typing import List

import json

from swift.plugin import ORM, orms
import sys
from pathlib import Path

import os

from swift.utils import get_logger
from swift.utils import (
    is_dist, is_mp, is_mp_ddp, get_dist_setting, get_device_count
)

entity_pool = []
entity_now = []

logger = get_logger()

freeze_env = os.getenv('FREEZE_NON_VISION_PARAMS')
logger.info(f"FREEZE_NON_VISION_PARAMS environment variable: {freeze_env}")

import os, torch


# 
def get_local_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def log_parallel_status(args):
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    ddp_avail = torch.distributed.is_available()
    ddp_inited = ddp_avail and torch.distributed.is_initialized()
    backend = None
    if ddp_inited:
        try:
            backend = torch.distributed.get_backend()
        except Exception:
            backend = "unknown"

    print(
        "[PARALLEL]",
        f"is_dist={is_dist()}",
        f"is_mp={is_mp()}",
        f"is_mp_ddp={is_mp_ddp()}",
        f"ddp_avail={ddp_avail}",
        f"ddp_inited={ddp_inited}",
        f"backend={backend}",
        f"RANK={rank}",
        f"LOCAL_RANK={local_rank}",
        f"WORLD_SIZE={world_size}",
        f"LOCAL_WORLD_SIZE={local_world_size}",
        f"n_gpu_visible={get_device_count()}",
        f"USE_FAST_INFERENCE={os.getenv('USE_FAST_INFERENCE', '0')}",
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
    )
    per_dev = getattr(args, "per_device_train_batch_size", None)
    ga = getattr(args, "gradient_accumulation_steps", None)
    dp_ws = world_size if is_dist() else 1
    if per_dev is not None and ga is not None:
        global_bsz = per_dev * ga * dp_ws
        print(f"[PARALLEL] per_device_bsz={per_dev} GA={ga} DP_world_size={dp_ws} => GLOBAL_BATCH={global_bsz}")


if os.getenv('FREEZE_NON_VISION_PARAMS') == '1':
    logger.info("Parameter freezing is ENABLED")
    from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer

    original_init = GRPOTrainer.__init__


    def patched_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)

        def freeze_parameters():
            frozen_count = 0
            trainable_count = 0
            trainable_params = []

            for name, param in self.model.named_parameters():
                # 只训练vision相关的LoRA参数和lm_head
                if any(pattern in name for pattern in [
                    'lora_A.vision', 'lora_B.vision'
                ]):
                    param.requires_grad = True
                    trainable_count += 1
                    trainable_params.append(name)
                    logger.info(f"[TRAINABLE] {name}")
                else:
                    param.requires_grad = False
                    frozen_count += 1

            logger.info(f"Frozen parameters: {frozen_count}")
            logger.info(f"Trainable parameters: {trainable_count}")
            logger.info(f"Trainable parameter list: {trainable_params}")

        if hasattr(self, 'model') and self.model is not None:
            freeze_parameters()

        try:
            log_parallel_status(self.args)
        except Exception as e:
            logger.warning(f"log_parallel_status failed: {e}")

        return result


    GRPOTrainer.__init__ = patched_init
else:
    logger.warning("Parameter freezing is DISABLED - FREEZE_NON_VISION_PARAMS not set to '1'")



def calculate_sentence_reward(prediction: str, ground_truth: str) -> float:
    """
    Calculates a binary reward based on *any* overlap between prediction and ground truth.

    Reward Rules:
    - If *at least one* predicted item matches a ground truth item, reward = 1.0.
    - If *no* predicted items match, reward = 0.0.
    - If ground truth is empty, reward = 1.0 if prediction is also empty, else 0.0.

    Args:
        prediction (str): A comma-separated string of predicted items.
        ground_truth (str): A comma-separated string of ground truth items.

    Returns:
        float: The calculated reward value (1.0 or 0.0).
    """
    # 1. Preprocessing: split, strip whitespace, convert to lowercase, and filter out empty strings.
    # (This logic is unchanged, it's correct)
    preds_list = [p.strip().lower() for p in prediction.split(',') if p.strip()]
    gts_list = [g.strip().lower() for g in ground_truth.split(',') if g.strip()]

    # 2. Convert to sets to ignore order and duplicates.
    # (This logic is unchanged)
    preds_set = set(preds_list)
    gts_set = set(gts_list)

    # Handle the edge case where the ground truth is empty.
    # (This logic is unchanged, it's robust)
    if not gts_set:
        return 1.0 if not preds_set else 0.0

    # 3. MODIFIED: Calculate reward based on the new "any match" rule.
    # We use isdisjoint() to check if the sets have *no* items in common.
    # If they are *not* disjoint (meaning they overlap by at least one), reward is 1.0.
    if not preds_set.isdisjoint(gts_set):
        return 1.0  # At least one match found
    else:
        return 0.0  # No matches found


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for idx, (content, sol) in enumerate(zip(completions, solution)):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))

            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


def kgkg(completions, solution) -> List[float]:
    rewards = []
    device = get_local_device()
    if '<think>' in solution[0]:
        an_match = re.search(r'<think>(.*?)</think>', str(solution[0]), re.DOTALL)
        ground_truth = an_match.group(1) if an_match else solution[0]
    else:
        ground_truth = solution[0]

    # completions.append(ground_truth)
    # from utils.get_knowledge_graph import extract_knowledge_graph
    # thinkkg = extract_knowledge_graph(completions)
    # 避免修改list_comple
    inputs = list(completions) + [ground_truth]
    from utils.get_knowledge_graph import extract_knowledge_graph
    with torch.cuda.device(device):
        thinkkg = extract_knowledge_graph(inputs, device=device)
    entity_set2 = set(thinkkg[-1]["entities"].items())
    relation_set2 = set(
        (rel["source_entity"], rel["target_entity"], rel["type"]) for rel in thinkkg[-1]["relations"])  # 关系-类别对

    thinkkg.pop()

    for idx, content in enumerate(thinkkg):
        entity_set1 = set(content["entities"].items())
        # 包含实体-类别对
        entity_over = entity_set1 & entity_set2
        print(f'################################## idx: {idx} ##################################')
        if idx == 0:
            print('solution kg: ', entity_set2)
        print('prediction kg: ', entity_set1)

        if len(entity_set1) == 0:
            precision = 0
        else:
            precision = len(entity_over) / len(entity_set1)

        # 计算召回率 (Recall)
        if len(entity_set2) == 0:
            recall = 0
        else:
            recall = len(entity_over) / len(entity_set2)

        # 计算F1分数
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        relation_set1 = set((rel["source_entity"], rel["target_entity"], rel["type"]) for rel in content["relations"])
        relation_over = relation_set1 & relation_set2

        if len(relation_set1) == 0:
            precision1 = 0
        else:
            precision1 = len(relation_over) / len(relation_set1)

        # 计算召回率 (Recall)
        if len(relation_set2) == 0:
            recall1 = 0
        else:
            recall1 = len(relation_over) / len(relation_set2)

        # 计算F1分数
        if precision1 + recall1 == 0:
            f = 0
        else:
            f = 2 * (precision1 * recall1) / (precision1 + recall1)
        rewards.append((f1 + f) / 2)
    print('kg rewards:', rewards)
    return rewards



class MultiModalAccuracyORM(ORM):
    def __init__(self, enable_debug=True):
        """
        Args:
            enable_debug (bool): If True, enable debug print statements.
        """
        self.enable_debug = enable_debug

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for idx, (content, sol) in enumerate(zip(completions, solution)):
            reward = 0.0
            if '<answer>' in sol:
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                cont_match = re.search(r"<answer>(.*?)</answer>", content)
                prediction = cont_match.group(1).strip() if cont_match else content.strip()

            else:
                pattern = r'[^.!?。！？]*(?=[.!?。！？]|$)'
                sol_match = re.search(pattern, sol)
                ground_truth = sol_match.group(0).strip() if sol_match else sol.strip()

                cont_match = re.search(pattern, content)
                prediction = cont_match.group(0).strip() if cont_match else content.strip()

            # Debug output if enabled
            if self.enable_debug:
                print(f'################################## idx: {idx} ##################################')
                if idx == 0:
                    print(f'solution: {sol}')
                    print(f'ground_truth: {ground_truth}\n')

                print(f'content: {content}')
                print(f'prediction: {prediction}\n')

            reward = calculate_sentence_reward(prediction, ground_truth)

            rewards.append(reward)

        # Debug output if enabled
        if self.enable_debug:
            print('Answer reward', rewards, '\n\n')
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class ReportKG_Jaccard(ORM):
    entity_pool = set()  # 类属性，所有实例共享
    entity_now = []

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        device = get_local_device()
        if '<think>' in solution[0]:
            an_match = re.search(r'<think>(.*?)</think>', str(solution[0]), re.DOTALL)
            ground_truth = an_match.group(1) if an_match else solution[0]
        else:
            ground_truth = solution[0]

        inputs = list(completions) + [ground_truth]
        from utils.get_knowledge_graph import extract_knowledge_graph
        with torch.cuda.device(device):
            thinkkg = extract_knowledge_graph(inputs, device=device)

        # 提取 Ground Truth (GT) 的实体和关系
        gt_kg = thinkkg[-1]
        entity_set2 = set(gt_kg["entities"].items())
        relation_set2 = set(
            (rel["source_entity"], rel["target_entity"], rel["type"]) for rel in gt_kg["relations"])

        self.__class__.entity_now = thinkkg.copy()
        thinkkg.pop()  # 移除 GT，只保留 completions 的 KG

        for idx, content in enumerate(thinkkg):
            # 1. 提取预测的实体和关系
            entity_set1 = set(content["entities"].items())
            relation_set1 = set(
                (rel["source_entity"], rel["target_entity"], rel["type"]) for rel in content["relations"])

            print(f'################################## idx: {idx} ##################################')
            if idx == 0:
                print('solution kg: ', entity_set2)
            print('prediction kg: ', entity_set1)

            # 2. 计算实体的 Jaccard (IoU) 分数
            entity_intersection = entity_set1 & entity_set2
            entity_union = entity_set1 | entity_set2

            if len(entity_union) == 0:
                # 如果并集为空（即 预测集 和 真实集 都为空），则认为完全匹配
                entity_score = 1.0
            else:
                entity_score = len(entity_intersection) / len(entity_union)

            # 3. 计算关系的 Jaccard (IoU) 分数
            relation_intersection = relation_set1 & relation_set2
            relation_union = relation_set1 | relation_set2

            if len(relation_union) == 0:
                # 如果并集为空（即 预测集 和 真实集 都为空），则认为完全匹配
                relation_score = 1.0
            else:
                relation_score = len(relation_intersection) / len(relation_union)

            # 4. 最终奖励是实体分数和关系分数的平均值
            final_reward = (entity_score + relation_score) / 2
            rewards.append(final_reward)

            # 打印调试信息，对比新旧分数
            print(
                f'Entity IoU: {entity_score:.4f}, Relation IoU: {relation_score:.4f}, Final Reward: {final_reward:.4f}')

        print('kg rewards:', rewards)
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_kg_jac'] = ReportKG_Jaccard