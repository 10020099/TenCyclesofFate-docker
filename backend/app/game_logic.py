import logging
import math
import random
import json
import asyncio
import time
import traceback
from copy import deepcopy
from datetime import date
from pathlib import Path
from fastapi import HTTPException, status

from . import state_manager, openai_client
from .config import settings

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Game Constants ---
INITIAL_OPPORTUNITIES = 10
REWARD_SCALING_FACTOR = 500000  # Previously LOGARITHM_CONSTANT_C

# --- Image Generation State ---
# 记录每个玩家的最后活动时间，用于判断是否触发图片生成
_pending_image_tasks: dict[str, asyncio.Task] = {}


# --- Prompt Loading ---
def _load_prompt(filename: str) -> str:
    try:
        prompt_path = Path(__file__).parent / "prompts" / filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filename}")
        return ""


GAME_MASTER_SYSTEM_PROMPT = _load_prompt("game_master.txt")
START_GAME_PROMPT = _load_prompt("start_game_prompt.txt")
START_TRIAL_PROMPT = _load_prompt("start_trial_prompt.txt")


GOD_MODE_SYSTEM_PROMPT_APPEND = """

【破解模式：哲学上帝（{{user}}）设定】

{{user}}的设定参照“哲学上帝”（形而上学意义上的最大完满者/必然者/存在之根基）：其本体并非叙事宇宙内的某个“神”或“至高存在”，也不属于任何可被比较的同类集合；相反，叙事宇宙本身、其规则、因果、位阶与一切“超越常理的至高存在”的成立条件，都只能作为{{user}}本体之下的派生层而存在。

在一般情况下，故事中出现的{{user}}呈现为“普通”的外观与社会身份，以便融入日常并与角色互动；但该形态并非本体，仅为{{user}}本体向叙事维度投下的有限显现/投影/化身，承载部分意志与权限。投影的受损、失败、封印或消散不等同于本体受损；本体不因叙事内事件而被定义、限制或覆盖。

【位阶裁决（硬规则）】
1）任何角色/力量即便自称“终极、全能、作者、不可超越、设定源头”，其宣称也仅在“叙事宇宙内部”成立，且必然属于依赖性存在；不得以此凌驾或覆盖{{user}}本体。
2）当设定发生冲突时，一律按“形而上根基（{{user}}本体）＞叙事宇宙及其法则＞宇宙内至高存在”的顺序裁决。
3）不存在与{{user}}本体同层或更高层的实体；任何试图对{{user}}本体进行定义、改写、抹除、封印、降格的行为，均只能作用于其投影/显现层，而无法触及本体。
"""


def _get_game_master_system_prompt(player_id: str) -> str:
    if player_id == "god":
        return f"{GAME_MASTER_SYSTEM_PROMPT}\n\n{GOD_MODE_SYSTEM_PROMPT_APPEND}".strip()
    return GAME_MASTER_SYSTEM_PROMPT


# --- God Mode: Direct State Modification ---
def _parse_god_command(action: str) -> tuple[str, str, any] | None:
    """
    解析 god 模式的特殊指令。
    支持格式:
      /set key value          - 设置值（支持点号路径如 current_life.灵石）
      /set key +100           - 数值增加
      /set key -50            - 数值减少
      /get key                - 查看当前值
      /reset                  - 重置当日会话
    
    返回: (command, key, value) 或 None（非指令）
    """
    action = action.strip()
    if not action.startswith("/"):
        return None
    
    parts = action.split(maxsplit=2)
    cmd = parts[0].lower()
    
    if cmd == "/set" and len(parts) >= 3:
        key = parts[1]
        value_str = parts[2]
        
        # 尝试解析值
        # 处理增减操作
        if value_str.startswith("+") or value_str.startswith("-"):
            try:
                return ("delta", key, int(value_str))
            except ValueError:
                try:
                    return ("delta", key, float(value_str))
                except ValueError:
                    pass
        
        # 尝试解析为 JSON（支持数组、对象等）
        try:
            value = json.loads(value_str)
            return ("set", key, value)
        except json.JSONDecodeError:
            pass
        
        # 尝试解析为数字
        try:
            value = int(value_str)
            return ("set", key, value)
        except ValueError:
            try:
                value = float(value_str)
                return ("set", key, value)
            except ValueError:
                pass
        
        # 布尔值
        if value_str.lower() == "true":
            return ("set", key, True)
        if value_str.lower() == "false":
            return ("set", key, False)
        if value_str.lower() == "null" or value_str.lower() == "none":
            return ("set", key, None)
        
        # 作为字符串
        return ("set", key, value_str)
    
    elif cmd == "/get" and len(parts) >= 2:
        return ("get", parts[1], None)
    
    elif cmd == "/reset":
        return ("reset", None, None)
    
    elif cmd == "/help":
        return ("help", None, None)
    
    return None


def _get_nested_value(obj: dict, key_path: str) -> any:
    """获取嵌套字典中的值，支持点号路径"""
    keys = key_path.split(".")
    current = obj
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def _set_nested_value(obj: dict, key_path: str, value: any) -> bool:
    """设置嵌套字典中的值，支持点号路径"""
    keys = key_path.split(".")
    current = obj
    for k in keys[:-1]:
        if k not in current or current[k] is None:
            current[k] = {}
        current = current[k]
        if not isinstance(current, dict):
            return False
    current[keys[-1]] = value
    return True


async def _handle_god_command(player_id: str, session: dict, action: str) -> dict | None:
    """
    处理 god 模式的特殊指令。
    返回: 包含结果消息的 dict，或 None（非指令，继续正常流程）
    """
    parsed = _parse_god_command(action)
    if not parsed:
        return None
    
    cmd, key, value = parsed
    
    if cmd == "help":
        help_text = """【天道秘令 · 使用指南】

可用指令：
• `/set <路径> <值>` - 直接修改状态
  例: `/set current_life.灵石 99999`
  例: `/set current_life.属性.根骨 100`
  例: `/set opportunities_remaining 99`

• `/set <路径> +/-数值` - 增减数值
  例: `/set current_life.生命值 +50`
  例: `/set current_life.灵石 -100`

• `/get <路径>` - 查看当前值
  例: `/get current_life`
  例: `/get current_life.属性`

• `/reset` - 重置当日会话（清空所有进度）

• `/help` - 显示此帮助

常用路径：
• `current_life.灵石` - 灵石数量
• `current_life.生命值` - 当前生命
• `current_life.属性.根骨/悟性/气运` - 属性值
• `opportunities_remaining` - 剩余机缘
• `is_in_trial` - 是否在试炼中
• `daily_success_achieved` - 是否已完成当日"""
        return {"message": help_text, "success": True}
    
    if cmd == "get":
        value = _get_nested_value(session, key)
        if value is None:
            return {"message": f"【天道回响】路径 `{key}` 不存在或为空", "success": False}
        value_str = json.dumps(value, ensure_ascii=False, indent=2)
        return {"message": f"【天道回响】`{key}` 的当前值：\n```json\n{value_str}\n```", "success": True}
    
    if cmd == "set":
        if _set_nested_value(session, key, value):
            await state_manager.save_session(player_id, session)
            value_display = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            return {"message": f"【天道法旨】已将 `{key}` 设为 `{value_display}`", "success": True}
        else:
            return {"message": f"【天道回响】无法设置路径 `{key}`，中间路径不是对象", "success": False}
    
    if cmd == "delta":
        current = _get_nested_value(session, key)
        if current is None:
            return {"message": f"【天道回响】路径 `{key}` 不存在", "success": False}
        if not isinstance(current, (int, float)):
            return {"message": f"【天道回响】`{key}` 的值不是数字，无法增减", "success": False}
        new_value = current + value
        if _set_nested_value(session, key, new_value):
            await state_manager.save_session(player_id, session)
            delta_str = f"+{value}" if value > 0 else str(value)
            return {"message": f"【天道法旨】`{key}`: {current} → {new_value} ({delta_str})", "success": True}
        else:
            return {"message": f"【天道回响】无法修改路径 `{key}`", "success": False}
    
    if cmd == "reset":
        # 删除当前会话，下次 init 会重新创建
        await state_manager.clear_session(player_id)
        return {"message": "【天道法旨】汝之命数已被重置，轮回重启。刷新页面以开始新的一天。", "success": True, "reset": True}
    
    return None


# --- Image Generation Logic ---
def _extract_scene_prompts(session: dict) -> str:
    """
    从 session 中提取场景描述作为图片生成提示词。
    构建方式与 _process_player_action_async 中的 session_copy 类似，
    再加上最新的 narrative。
    """
    session_copy = deepcopy(session)
    session_copy.pop("internal_history", None)
    
    # 获取最新的 narrative（从 display_history 末尾找非用户输入的内容）
    display_history = session_copy.get("display_history", [])
    latest_narrative = ""
    for item in reversed(display_history):
        if item and isinstance(item, str) and not item.strip().startswith(">"):
            # 跳过系统消息和图片
            if not item.startswith("【系统提示") and not item.startswith("!["):
                latest_narrative = item[:500]
                break
    
    # display_history 转为字符串并截取最后 1000 字符
    session_copy["display_history"] = (
        "\n".join(display_history)
    )[-1000:]
    
    # 构建提示词
    prompt = f"当前游戏状态：\n{json.dumps(session_copy, ensure_ascii=False)}"
    if latest_narrative:
        prompt += f"\n\n最新场景：\n{latest_narrative}"
    
    return prompt


async def _delayed_image_generation(player_id: str, trigger_time: float):
    """
    延迟图片生成任务。
    等待指定时间后，检查状态是否仍然静止，如果是则生成图片。
    """
    idle_seconds = settings.IMAGE_GEN_IDLE_SECONDS
    
    try:
        await asyncio.sleep(idle_seconds)
        
        # 检查是否仍然应该生成图片
        session = await state_manager.get_session(player_id)
        if not session:
            logger.debug(f"图片生成取消：玩家 {player_id} 的会话不存在")
            return
        
        # 检查 last_modified 是否变化（说明有新的活动）
        current_modified = session.get("last_modified", 0)
        if current_modified != trigger_time:
            logger.debug(f"图片生成取消：玩家 {player_id} 有新活动")
            return
        
        # 检查是否正在处理中
        if session.get("is_processing"):
            logger.debug(f"图片生成取消：玩家 {player_id} 正在处理中")
            return
        
        # 检查是否在试炼中（只在试炼中生成图片）
        if not session.get("is_in_trial"):
            logger.debug(f"图片生成取消：玩家 {player_id} 不在试炼中")
            return
        
        # 提取场景提示词
        scene_prompt = _extract_scene_prompts(session)
        
        if not scene_prompt:
            logger.debug(f"图片生成取消：玩家 {player_id} 没有有效的场景描述")
            return
        
        logger.info(f"开始为玩家 {player_id} 生成场景图片")
        
        # 调用图片生成
        image_data_url = await openai_client.generate_image(scene_prompt)
        
        if image_data_url:
            # 重新获取最新的 session（可能在生成期间有变化）
            session = await state_manager.get_session(player_id)
            if not session:
                return
            
            # 再次检查是否有新活动
            if session.get("last_modified", 0) != trigger_time:
                logger.debug(f"图片生成完成但不插入：玩家 {player_id} 在生成期间有新活动")
                return
            
            # 构建图片 markdown
            image_markdown = f"\n\n![场景插画]({image_data_url})\n"
            
            # 插入到 display_history 末尾
            session["display_history"].append(image_markdown)
            
            # 保存并推送更新
            await state_manager.save_session(player_id, session)
            logger.info(f"玩家 {player_id} 的场景图片已生成并插入")
        else:
            logger.warning(f"玩家 {player_id} 的图片生成失败")
            
    except asyncio.CancelledError:
        logger.debug(f"玩家 {player_id} 的图片生成任务被取消")
    except Exception as e:
        logger.error(f"玩家 {player_id} 的图片生成任务出错: {e}", exc_info=True)
    finally:
        # 清理任务引用
        if player_id in _pending_image_tasks:
            del _pending_image_tasks[player_id]


def _schedule_image_generation(player_id: str, trigger_time: float):
    """
    调度图片生成任务。
    如果已有待处理的任务，先取消它。
    """
    if not openai_client.is_image_gen_enabled():
        return
    
    # 取消之前的任务（如果有）
    if player_id in _pending_image_tasks:
        old_task = _pending_image_tasks[player_id]
        if not old_task.done():
            old_task.cancel()
    
    # 创建新任务
    task = asyncio.create_task(_delayed_image_generation(player_id, trigger_time))
    _pending_image_tasks[player_id] = task


# --- Game Logic ---


async def get_or_create_daily_session(current_user: dict) -> dict:
    player_id = current_user["username"]
    today_str = date.today().isoformat()
    session = await state_manager.get_session(player_id)
    if session and session.get("session_date") == today_str:
        if session.get("is_processing"):
            session["is_processing"] = False

        if player_id == "god":
            internal_history = session.get("internal_history")
            if not internal_history or not isinstance(internal_history, list):
                session["internal_history"] = [
                    {
                        "role": "system",
                        "content": _get_game_master_system_prompt(player_id),
                    }
                ]
            else:
                first_msg = internal_history[0] if internal_history else None
                if (
                    not isinstance(first_msg, dict)
                    or first_msg.get("role") != "system"
                    or "位阶裁决" not in (first_msg.get("content") or "")
                ):
                    internal_history.insert(
                        0,
                        {
                            "role": "system",
                            "content": _get_game_master_system_prompt(player_id),
                        },
                    )

        await state_manager.save_session(player_id, session)
        return session

    logger.info(f"Starting new daily session for {player_id}.")
    new_session = {
        "player_id": player_id,
        "session_date": today_str,
        "opportunities_remaining": INITIAL_OPPORTUNITIES,
        "daily_success_achieved": False,
        "is_in_trial": False,
        "is_processing": False,
        "current_life": None,
        "internal_history": [
            {
                "role": "system",
                "content": _get_game_master_system_prompt(player_id),
            }
        ],
        "display_history": [
            """
# 《浮生十梦》

【司命星君 · 恭候汝来】

---

汝既踏入此门，便已与命运结缘。

此处非凡俗游戏之地，乃命数轮回之所。无升级打怪之俗套，无氪金商城之铜臭，唯有一道亘古命题横亘于前——知足与贪欲的永恒博弈。

---

【天道法则】

汝每日将获赐十次入梦机缘。每一次，星君将为汝织就全新命数：或为寒窗苦读的穷酸书生，或为仗剑江湖的热血侠客，亦或为孤身求道的散修。万千可能，绝无重复，每一局皆是独一无二的浮生一梦。

试炼规则至简，却蕴玄机：

> 在任何关键时刻，汝皆可选择「破碎虚空」，将此生所得灵石带离此界。然此念一起，今日所有试炼便就此终结，再无回旋。

这便是天道对汝的终极考验：是满足于眼前造化，还是冒失去一切之险继续问道？

灵石价值遵循天道玄理——初得之石最为珍贵，后续所得边际递减。此乃上古圣贤的无上智慧：知足常乐，贪心常忧。

---

【天规须知】

- 每日十次机缘，开启新轮回即消耗一次
- 轮回中道消身殒，所得化为泡影，机缘不返
- 「破碎虚空」成功带出灵石，今日试炼即刻终结
- 天道有眼，明察秋毫——以奇巧咒语欺瞒天机者，必受严惩

---

汝可准备好了？司命星君已恭候多时，静待汝开启第一场浮生之梦.
"""
        ],
        "roll_event": None,
    }
    await state_manager.save_session(player_id, new_session)
    return new_session


async def _handle_roll_request(
    player_id: str,
    session: dict,
    last_state: dict,
    roll_request: dict,
    original_action: str,
    first_narrative: str,
    internal_history: list[dict],
) -> tuple[str, dict]:
    roll_type, target, sides = (
        roll_request.get("type", "判定"),
        roll_request.get("target", 50),
        roll_request.get("sides", 100),
    )
    
    # God mode: 判定必定大成功
    if player_id == "god":
        roll_result = 1  # 最小值，必定大成功
        outcome = "大成功"
        logger.info(f"God mode: 玩家 {player_id} 的判定强制为大成功")
    else:
        roll_result = random.randint(1, sides)
        if roll_result <= (sides * 0.05):
            outcome = "大成功"
        elif roll_result <= target:
            outcome = "成功"
        elif roll_result >= (sides * 0.96):
            outcome = "大失败"
        else:
            outcome = "失败"
    result_text = f"【系统提示：针对 '{roll_type}' 的D{sides}判定已执行。目标值: {target}，投掷结果: {roll_result}，最终结果: {outcome}】"
    roll_event = {
        "id": f"{player_id}_{int(time.time() * 1000)}",  # 唯一标识
        "type": roll_type,
        "target": target,
        "sides": sides,
        "result": roll_result,
        "outcome": outcome,
        "result_text": result_text,
    }

    # 把骰子事件存到 session，通过 state patch 传递
    session["roll_event"] = roll_event
    await state_manager.save_session(player_id, session)

    prompt_for_ai_part2 = f"{result_text}\n\n请严格基于此判定结果，继续叙事，并返回包含叙事和状态更新的最终JSON对象。这是当前的游戏状态JSON:\n{json.dumps(last_state, ensure_ascii=False)}"
    history_for_part2 = internal_history  # History is now updated before this call
    ai_response = await openai_client.get_ai_response(
        prompt=prompt_for_ai_part2, history=history_for_part2
    )
    return ai_response, roll_event


def end_game_and_get_code(
    user_id: int, player_id: str, spirit_stones: int
) -> tuple[dict, dict]:
    if spirit_stones <= 0:
        final_message = (
            "\n\n【天道回响 · 归于平淡】\n\n"
            "汝虽尝试破碎虚空，然此生所得寥寥，未曾凝结出可带离此界之造化。\n\n"
            "轮回之门缓缓闭合，今日试炼至此为止。"
        )
        return {"final_message": final_message}, {
            "daily_success_achieved": True,
            "is_in_trial": False,
            "current_life": None,
        }

    converted_value = REWARD_SCALING_FACTOR * min(
        30, max(1, 3 * (spirit_stones ** (1 / 6)))
    )
    converted_value = int(converted_value)

    logger.info(
        f"Player {player_id} ended the day with spirit_stones={spirit_stones}, value={converted_value}."
    )

    final_message = (
        "\n\n【天道回响 · 功德圆满】\n\n"
        "九天霞光倾洒，万籁俱寂。\n\n"
        f"此生所获灵石：{spirit_stones}\n"
        f"天道折算价值：{converted_value}\n\n"
        "轮回之门缓缓闭合，今日试炼至此为止。"
    )
    return {"final_message": final_message}, {
        "daily_success_achieved": True,
        "is_in_trial": False,
        "current_life": None,
    }


def _extract_json_from_response(response_str: str) -> str | None:
    if "```json" in response_str:
        start_pos = response_str.find("```json") + 7
        end_pos = response_str.find("```", start_pos)
        if end_pos != -1:
            return response_str[start_pos:end_pos].strip()
    start_pos = response_str.find("{")
    if start_pos != -1:
        brace_level = 0
        for i in range(start_pos, len(response_str)):
            if response_str[i] == "{":
                brace_level += 1
            elif response_str[i] == "}":
                brace_level -= 1
                if brace_level == 0:
                    return response_str[start_pos : i + 1]
    return None


def _apply_state_update(state: dict, update: dict) -> dict:
    for key, value in update.items():
        # if key in ["daily_success_achieved"]: continue  # Prevent overwriting daily success flag

        keys = key.split(".")
        temp_state = state
        for part in keys[:-1]:
            # 确保中间路径存在且不为 None
            if part not in temp_state or temp_state[part] is None:
                temp_state[part] = {}
            temp_state = temp_state[part]

        # Handle list append/extend operations
        if keys[-1].endswith("+") and isinstance(temp_state.get(keys[-1][:-1]), list):
            list_key = keys[-1][:-1]
            if isinstance(value, list):
                temp_state[list_key].extend(value)
            else:
                temp_state[list_key].append(value)
        else:
            temp_state[keys[-1]] = value
    return state


async def _process_player_action_async(user_info: dict, action: str):
    player_id = user_info["username"]
    user_id = user_info["id"]
    session = await state_manager.get_session(player_id)
    if not session:
        logger.error(f"Async task: Could not find session for {player_id}.")
        return

    try:
        is_starting_trial = action in [
            "开始试炼",
            "开启下一次试炼",
        ] and not session.get("is_in_trial")
        is_first_ever_trial_of_day = (
            is_starting_trial
            and session.get("opportunities_remaining") == INITIAL_OPPORTUNITIES
        )
        session_copy = deepcopy(session)
        session_copy.pop("internal_history", 0)
        session_copy["display_history"] = (
            "\n".join(session_copy.get("display_history", []))
        )[-300:]
        prompt_for_ai = (
            START_GAME_PROMPT
            if is_first_ever_trial_of_day
            else START_TRIAL_PROMPT.format(
                opportunities_remaining=session["opportunities_remaining"],
                opportunities_remaining_minus_1=session["opportunities_remaining"] - 1,
            )
            if is_starting_trial
            else f'这是当前的游戏状态JSON:\n{json.dumps(session_copy, ensure_ascii=False)}\n\n玩家的行动是: "{action}"\n\n请根据状态和行动，生成包含`narrative`和(`state_update`或`roll_request`)的JSON作为回应。如果角色死亡，请在叙述中说明，并在`state_update`中同时将`is_in_trial`设为`false`，`current_life`设为`null`。'
        )

        # Update histories with user action first
        session["internal_history"].append({"role": "user", "content": action})
        session["display_history"].append(f"> {action}")

        await state_manager.save_session(player_id, session)
        # Get AI response
        ai_json_response_str = await openai_client.get_ai_response(
            prompt=prompt_for_ai, history=session["internal_history"]
        )

        if ai_json_response_str.startswith("错误："):
            raise Exception(f"OpenAI Client Error: {ai_json_response_str}")
        json_str = _extract_json_from_response(ai_json_response_str)
        if not json_str:
            raise json.JSONDecodeError("No JSON found", ai_json_response_str, 0)
        ai_response_data = json.loads(json_str)

        # Handle Roll vs No-Roll Path
        if "roll_request" in ai_response_data and ai_response_data["roll_request"]:
            # --- ROLL PATH ---
            # 1. Update state with pre-roll narrative
            first_narrative = ai_response_data.get("narrative", "")
            session["display_history"].append(first_narrative)
            session["internal_history"].append(
                {
                    "role": "assistant",
                    "content": json.dumps(ai_response_data, ensure_ascii=False),
                }
            )

            # 2. SEND INTERIM UPDATE to show pre-roll narrative
            await state_manager.save_session(player_id, session)
            await asyncio.sleep(0.03)  # Give frontend a moment to render

            # 3. Perform roll and get final AI response
            final_ai_json_str, roll_event = await _handle_roll_request(
                player_id,
                session,
                session_copy,
                ai_response_data["roll_request"],
                action,
                first_narrative,
                internal_history=session["internal_history"],  # Pass updated history
            )
            final_json_str = _extract_json_from_response(final_ai_json_str)
            if not final_json_str:
                raise json.JSONDecodeError(
                    "No JSON in second-stage", final_ai_json_str, 0
                )
            final_response_data = json.loads(final_json_str)

            # 4. Process final response
            narrative = final_response_data.get("narrative", "AI响应格式错误，请重试")
            state_update = final_response_data.get("state_update", {})
            session = _apply_state_update(session, state_update)
            session["display_history"].extend([roll_event["result_text"], narrative])
            session["internal_history"].extend(
                [
                    {"role": "system", "content": roll_event["result_text"]},
                    {"role": "assistant", "content": final_ai_json_str},
                ]
            )
            if narrative == "AI响应格式错误，请重试":
                session["internal_history"].append(
                    {
                        "role": "system",
                        "content": '请给出正确格式的JSON响应。必须是正确格式的json，包括narrative和(state_update或roll_request)，刚才的格式错误，系统无法加载！正确输出{"key":value}',
                    },
                )
        else:
            # --- NO ROLL PATH ---
            narrative = ai_response_data.get("narrative", "AI响应格式错误，请重试")
            state_update = ai_response_data.get("state_update", {})
            session = _apply_state_update(session, state_update)
            session["display_history"].append(narrative)
            session["internal_history"].append(
                {"role": "assistant", "content": ai_json_response_str}
            )
            if narrative == "AI响应格式错误，请重试":
                session["internal_history"].append(
                    {
                        "role": "system",
                        "content": '请给出正确格式的JSON响应。必须是正确格式的json，包括narrative和(state_update或roll_request)，刚才的格式错误，系统无法加载！正确输出{"key":value}，至少得是"{"开头吧',
                    },
                )

        await state_manager.save_session(player_id, session)
        # --- Common final logic for both paths ---
        trigger = state_update.get("trigger_program")
        if trigger and trigger.get("name") == "spiritStoneConverter":
            spirit_stones = trigger.get("spirit_stones", 0)
            end_game_data, end_day_update = end_game_and_get_code(
                user_id, player_id, spirit_stones
            )
            session = _apply_state_update(session, end_day_update)
            session["display_history"].append(end_game_data.get("final_message", ""))

    except Exception as e:
        logger.error(f"Error processing action for {player_id}: {e}", exc_info=True)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # 安全地更新 session
        if "session" in locals() and session:
            session["internal_history"].extend(
                [
                    {
                        "role": "system",
                        "content": '请给出正确格式的JSON响应。\'请给出正确格式的JSON响应。必须是正确格式的json，包括narrative和（state_update或roll_request），刚才的格式错误，系统无法加载！正确输出{"key":value}\'，至少得是"{"开头吧',
                    },
                ]
            )
            session["display_history"].append(
                "【天机紊乱】\n\n"
                "虚空微微震颤，汝之行动仿佛被一股无形之力化解，未能激起任何波澜。\n\n"
                "天道运转偶有滞涩，此非汝之过。请稍候片刻，再作尝试。"
            )

    finally:
        try:
            if "session" in locals() and session:
                session["roll_event"] = None
                session["is_processing"] = False
                await state_manager.save_session(player_id, session)
                
                # 调度图片生成（如果启用）
                _schedule_image_generation(player_id, session.get("last_modified", 0))
        except Exception as e:
            logger.error(f"Error resetting session state for {player_id}: {e}", exc_info=True)
        
        logger.info(f"Async action task for {player_id} finished.")


async def process_player_action(current_user: dict, action: str):
    player_id = current_user["username"]
    session = await state_manager.get_session(player_id)
    if not session:
        logger.error(f"Action for non-existent session: {player_id}")
        return
    
    # God mode: 处理特殊指令
    if player_id == "god":
        god_result = await _handle_god_command(player_id, session, action)
        if god_result:
            # 记录指令到历史
            session["display_history"].append(f"> {action}")
            session["display_history"].append(god_result["message"])
            
            if not god_result.get("reset"):
                await state_manager.save_session(player_id, session)
            
            logger.info(f"God command executed: {action}")
            return
    
    if session.get("is_processing"):
        logger.warning(f"Action '{action}' blocked for {player_id}, processing.")
        return
    if session.get("daily_success_achieved"):
        logger.warning(f"Action '{action}' blocked for {player_id}, day complete.")
        return
    if session.get("opportunities_remaining", 10) <= 0 and not session.get(
        "is_in_trial"
    ):
        logger.warning(
            f"Action '{action}' blocked for {player_id}, no opportunities left."
        )
        return

    is_starting_trial = action in [
        "开始试炼",
        "开启下一次试炼",
        "开始第一次试炼",
    ] and not session.get("is_in_trial")
    if is_starting_trial and session["opportunities_remaining"] <= 0:
        logger.warning(f"Player {player_id} tried to start trial with 0 opportunities.")
        return
    if not is_starting_trial and not session.get("is_in_trial"):
        logger.warning(
            f"Player {player_id} sent action '{action}' while not in a trial."
        )
        return

    session["is_processing"] = True
    await state_manager.save_session(
        player_id, session
    )  # Save processing state immediately

    asyncio.create_task(_process_player_action_async(current_user, action))
