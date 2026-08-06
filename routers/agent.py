from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from agents.registry import tool_registry
from agents.runner import run_single_tool_call
from database import get_db
from schemas import ToolCallingRequest


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


@router.post("/tool-call")
def tool_call(
    req: ToolCallingRequest,
    db: Session = Depends(get_db),
):
    """
    单轮 Tool Calling 测试接口。
    """

    result = run_single_tool_call(
        user_message=req.message,
        db=db,
        temperature=req.temperature,
        tool_choice=req.tool_choice,
    )

    return {
        "code": 200,
        "message": "success",
        "data": result,
    }