import asyncio
import json
import queue
import re

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agents.graph_agent import run_agent_graph
from agents.registry import tool_registry
from database import SessionLocal, get_db
from models import ChatHistory
from schemas import AgentRunRequest


# 实质内容 token：汉字 / 字母 / 数字。
# 去掉这些以外的字符后为空，说明消息只有标点 / 空白。
_MEANINGFUL_PATTERN = re.compile(
    r"[^\u4e00-\u9fa5A-Za-z0-9]"
)

# 无实质内容输入时的引导语
_EMPTY_GUIDE = (
    "我没有收到具体的问题内容。请问你想了解什么？\n"
    "你可以直接描述问题，例如：\n"
    "- 这篇文档主要讲了什么？\n"
    "- 模型用了什么优化器？\n"
    "- 两份文档有什么区别？"
)


def _is_empty_message(message: str | None) -> bool:
    """判断消息是否为纯标点 / 空白、无实质内容。"""
    if not message:
        return True
    return len(_MEANINGFUL_PATTERN.sub("", message)) == 0


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


@router.get("/tools")
def list_tools():
    """
    查看当前系统已经注册的工具。
    """

    return {
        "code": 200,
        "message": "success",
        "data": {
            "tool_names": (
                tool_registry.get_tool_names()
            ),
            "tools": (
                tool_registry.get_api_schemas()
            ),
        },
    }


@router.post("/graph-run")
def run_agent_graph_endpoint(
    req: AgentRunRequest,
    db: Session = Depends(get_db),
):
    """
    多步骤 Agent 执行接口（LangGraph 版本，非流式）。
    """

    result = run_agent_graph(
        user_message=req.message,
        db=db,
        max_steps=req.max_steps,
        history=getattr(req, "history", None),
    )

    return {
        "code": 200,
        "message": "success",
        "data": result,
    }


def format_sse(data: dict) -> str:
    """把字典转成 SSE 格式。"""
    json_data = json.dumps(data, ensure_ascii=False, default=str)
    return f"data: {json_data}\n\n"


# 后台保存任务引用集合，避免任务被 GC 回收
_background_tasks = set()


def schedule_history_save(message, answer, session_id, tool_name, tool_call_count, duration_ms, citations=None):
    """把对话历史保存放到后台任务，绝不阻塞 SSE 流关闭。"""
    citations_text = None
    if citations:
        try:
            citations_text = json.dumps(
                citations, ensure_ascii=False
            )
        except Exception:
            citations_text = None

    async def _run():
        save_db = SessionLocal()
        try:
            save_db.add(ChatHistory(
                user_message=message,
                ai_answer=answer,
                session_id=session_id,
                tool_name=tool_name,
                tool_call_count=tool_call_count,
                duration_ms=duration_ms,
                citations_json=citations_text,
            ))
            # 同步 commit 放线程池，避免卡住事件循环
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_db.commit)
        except Exception as e:
            print(f"保存对话历史失败: {e}")
            try:
                save_db.rollback()
            except Exception:
                pass
        finally:
            save_db.close()

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@router.post("/graph-run-stream")
async def run_agent_graph_stream(
    req: AgentRunRequest,
    db: Session = Depends(get_db),
):
    """
    Agent 流式执行接口（LangGraph 版本）。

    SSE 事件类型：
    - meta: 开始执行
    - step: 每一步的信息（工具调用等）
    - answer_chunk: 最终回答的逐字片段
    - citations: 引用来源
    - done: 执行完成
    - error: 出错
    """

    async def generate():
        # 推送开始事件
        yield format_sse({
            "type": "meta",
            "message": "agent started",
        })

        # ------------------------------------------------
        # 纯标点 / 空白输入：不启动 graph、不调用检索工具，
        # 直接流式返回引导语
        # ------------------------------------------------
        if _is_empty_message(req.message):
            # 按标点分片，模拟流式
            chunks = re.split(r"([\n。？！])", _EMPTY_GUIDE)
            guide = ""
            for piece in chunks:
                if piece:
                    guide += piece
                    yield format_sse({
                        "type": "answer_chunk",
                        "content": piece,
                    })

            yield format_sse({
                "type": "done",
                "tool_call_count": 0,
                "total_duration_ms": 0,
            })
            schedule_history_save(
                req.message,
                guide,
                getattr(req, "session_id", None),
                None,
                0,
                0,
                [],
            )
            return

        # 线程安全队列：graph 在 executor 线程里把 step / token 推入，
        # 本协程从中取出并实时 yield（真流式）
        event_queue: queue.Queue = queue.Queue()

        def on_token(delta: str) -> None:
            event_queue.put(("token", delta))

        def on_step(trace: dict) -> None:
            event_queue.put(("step", trace))

        def run_with_sentinel():
            try:
                return run_agent_graph(
                    user_message=req.message,
                    db=db,
                    max_steps=req.max_steps,
                    history=getattr(req, "history", None),
                    on_token=on_token,
                    on_step=on_step,
                )
            finally:
                # 无论成功或异常，都放入结束标记
                event_queue.put((None, None))

        loop = asyncio.get_event_loop()
        graph_future = loop.run_in_executor(
            None, run_with_sentinel
        )

        answer = ""

        try:
            # 边执行 graph，边实时转发 step / token
            while True:
                kind, payload = await loop.run_in_executor(
                    None, event_queue.get
                )

                if kind is None:
                    # graph 已结束
                    break

                if kind == "step":
                    yield format_sse({
                        "type": "step",
                        "step": payload,
                    })
                elif kind == "token":
                    answer += payload
                    yield format_sse({
                        "type": "answer_chunk",
                        "content": payload,
                    })

            # 取 graph 完整结果（可能在此抛异常）
            result = await graph_future

            # 从 traces 提取引用来源
            citations = []
            for trace in result.get("traces", []):
                if trace.get("tool_name") == "ask_advanced_rag":
                    tool_result = trace.get("result", {})
                    if (
                        isinstance(tool_result, dict)
                        and tool_result.get("citations")
                    ):
                        citations = tool_result["citations"]
                elif trace.get("tool_name") == "search_knowledge_base":
                    tool_result = trace.get("result", {})
                    if (
                        isinstance(tool_result, dict)
                        and tool_result.get("results")
                    ):
                        citations = [
                            {
                                "citation_id": r.get("rank", i + 1),
                                "file_id": tool_result.get("file_id"),
                                "filename": tool_result.get("filename", ""),
                                "chunk_id": r.get("chunk_id"),
                                "content": r.get("content", ""),
                                "rerank_score": None,
                            }
                            for i, r in enumerate(tool_result["results"])
                        ]

            # 正文推送完成后再推送引用来源
            if citations:
                yield format_sse({
                    "type": "citations",
                    "citations": citations,
                })

            # 提取元信息
            session_id = getattr(req, "session_id", None)
            tool_name = None
            for trace in result.get("traces", []):
                if trace.get("tool_name"):
                    tool_name = trace["tool_name"]
                    break

            # 先推送 done，让前端立即复位
            yield format_sse({
                "type": "done",
                "tool_call_count": result.get("tool_call_count", 0),
                "total_duration_ms": result.get(
                    "timing", {}
                ).get("total_duration_ms", 0),
            })

            # 保存历史丢给后台任务
            schedule_history_save(
                req.message,
                answer,
                session_id,
                tool_name,
                result.get("tool_call_count", 0),
                result.get("timing", {}).get("total_duration_ms", 0),
                citations,
            )

        except Exception as e:
            yield format_sse({
                "type": "error",
                "message": str(e),
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
