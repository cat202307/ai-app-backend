from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session


class BaseTool(ABC):
    """
    所有 Agent 工具的统一基类。

    每个工具必须声明：

    1. name
       工具唯一名称。

    2. description
       告诉模型工具做什么、什么时候应该使用。

    3. input_model
       使用 Pydantic 定义工具参数。

    4. execute()
       真实执行工具逻辑。
    """

    name: str
    description: str
    input_model: type[BaseModel]

    def to_api_schema(self) -> dict[str, Any]:
        """
        将工具转换为 DeepSeek/OpenAI 兼容的 tools Schema。
        """

        parameters = self.input_model.model_json_schema()

        # 不允许模型额外生成未定义参数。
        parameters["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @abstractmethod
    def execute(
        self,
        arguments: BaseModel,
        db: Session,
    ) -> dict[str, Any]:
        """
        执行真实工具逻辑。
        """

        raise NotImplementedError