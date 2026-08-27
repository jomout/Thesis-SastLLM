from typing import Any

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from argus.configs import get_logger

logger = get_logger(__name__)


class CustomLLM(BaseChatModel):
    """Custom chat model wrapper for Endpoint on GCP."""

    endpoint_url: str
    access_token: str

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages → API schema
        payload_msgs = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            payload_msgs.append({"role": role, "content": msg.content})

        payload = {"messages": payload_msgs}
        headers = {"access_token": self.access_token}

        logger.debug("Sending custom LLM request", endpoint=self.endpoint_url, messages=len(payload_msgs))
        try:
            resp = requests.post(self.endpoint_url, json=payload, headers=headers)
            resp.raise_for_status()
        except requests.RequestException as error:
            logger.exception("Custom LLM request failed", endpoint=self.endpoint_url, error=str(error))
            raise
        data = resp.json()
        logger.debug("Custom LLM request completed", endpoint=self.endpoint_url, status_code=resp.status_code)

        content = str(data)

        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "endpoint_url": self.endpoint_url,
            "model_name": "CustomLLM",
        }

    @property
    def _llm_type(self) -> str:
        return "custom-llm"
