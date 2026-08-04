from sqlalchemy.orm import Session

from core.config import settings
from models import ChatHistory
from services.prompt_service import build_chat_system_message


def load_recent_history(
    db: Session,
    session_id: str | None,
    limit: int | None = None,
    max_chars: int | None = None,
) -> list[dict[str, str]]:
    """
    加载当前会话最近的聊天历史，并转换为模型消息格式。

    参数：
        db:
            SQLAlchemy 数据库 Session。

        session_id:
            当前会话 ID。
            如果为空，表示本次聊天不需要历史上下文。

        limit:
            最多读取多少条 ChatHistory。
            如果不传，使用配置 CHAT_HISTORY_LIMIT。

        max_chars:
            历史内容最多允许多少字符。
            如果不传，使用配置 CHAT_HISTORY_MAX_CHARS。

    返回：
        OpenAI / DeepSeek messages 格式：

        [
            {
                "role": "user",
                "content": "第一个问题"
            },
            {
                "role": "assistant",
                "content": "第一个回答"
            }
        ]

    说明：
        数据库查询按时间倒序，先得到最新记录。
        最终发送给模型前，会重新转换成旧到新的顺序。
    """

    if not session_id:
        return []

    actual_limit = (
        limit
        if limit is not None
        else settings.chat_history_limit
    )

    actual_max_chars = (
        max_chars
        if max_chars is not None
        else settings.chat_history_max_chars
    )

    if actual_limit <= 0:
        return []

    records = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.session_id == session_id
        )
        .order_by(
            ChatHistory.created_at.desc()
        )
        .limit(actual_limit)
        .all()
    )

    # records 当前顺序：
    # 最新记录 → 最旧记录
    #
    # 从最新记录开始选择，
    # 确保超过字符预算时优先保留最近对话。
    selected_records: list[ChatHistory] = []
    total_chars = 0

    for record in records:
        user_message = record.user_message or ""
        ai_answer = record.ai_answer or ""

        record_chars = (
            len(user_message)
            + len(ai_answer)
        )

        # 已经选中过记录，并且加入当前记录会超过预算时，
        # 停止继续添加更早的历史。
        if (
            selected_records
            and total_chars + record_chars > actual_max_chars
        ):
            break

        selected_records.append(record)
        total_chars += record_chars

    # 模型消息必须按照：
    # 旧对话 → 新对话
    selected_records.reverse()

    messages: list[dict[str, str]] = []

    for record in selected_records:
        if record.user_message:
            messages.append({
                "role": "user",
                "content": record.user_message,
            })

        if record.ai_answer:
            messages.append({
                "role": "assistant",
                "content": record.ai_answer,
            })

    return messages


def build_chat_messages(
    db: Session,
    user_message: str,
    session_id: str | None = None,
) -> list[dict[str, str]]:
    """
    构造一次普通聊天需要发送给模型的完整 messages。

    组成部分：

        System Prompt
        + 最近历史对话
        + 用户当前问题

    当前阶段还没有加入：

        - 历史摘要
        - RAG 检索内容
        - 工具执行结果
        - 精确 Token 预算

    这些能力会在后续继续扩展。
    """

    messages: list[dict[str, str]] = [
        build_chat_system_message()
    ]

    history_messages = load_recent_history(
        db=db,
        session_id=session_id,
    )

    messages.extend(history_messages)

    messages.append({
        "role": "user",
        "content": user_message,
    })

    return messages


def get_context_statistics(
    messages: list[dict[str, str]],
) -> dict[str, int]:
    """
    统计本次上下文的基础信息。

    当前使用字符数量做简单统计，
    后续再替换为模型 Token 统计器。
    """

    total_characters = sum(
        len(message.get("content", ""))
        for message in messages
    )

    user_messages = sum(
        1
        for message in messages
        if message.get("role") == "user"
    )

    assistant_messages = sum(
        1
        for message in messages
        if message.get("role") == "assistant"
    )

    system_messages = sum(
        1
        for message in messages
        if message.get("role") == "system"
    )

    return {
        "message_count": len(messages),
        "character_count": total_characters,
        "system_message_count": system_messages,
        "user_message_count": user_messages,
        "assistant_message_count": assistant_messages,
    }