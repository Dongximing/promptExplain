from sammo.components import GenerateText, Output
from sammo.utils import serialize_json

from sammo.base import LLMResult, Costs
from sammo.runners import BaseRunner
import getpass
import ssl
class DeepInfraChat(BaseRunner):
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float | None = 0,
        seed: int = 0,
        priority: int = 0,
        **kwargs,
    ) -> LLMResult:
        formatted_prompt = f"{prompt}"
        request = dict(
            inputs=formatted_prompt,
        )
        fingerprint = serialize_json({"seed": seed, "generative_model_id": self._model_id, **request})
        return await self._execute_request(request, fingerprint, priority)

    async def _call_backend(self, request: dict) -> dict:
        async with self._get_session() as session:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            async with session.post(
                url="http://localhost:8082",
                json=request,
                headers={"Authorization":""},

            ) as response:
                if response.status == 422:
                    print("Error 422: Unprocessable Entity. Check data format and required fields.")
                    text = await response.text()
                    print("Response from server:", text)
                print("json_data------->\n", response.json())
                return await response.json()

    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes):
        print("json_data------->\n",json_data)
        return LLMResult(
            json_data["results"][0]["generated_text"],
            costs=Costs(json_data["num_input_tokens"], json_data["num_tokens"]),
        )
runner = DeepInfraChat(
    "meta-llama/Llama-2-7b-hf", api_config={"api_key": ""}
)
print(Output(GenerateText("What is the capital of China? the answer is")).run(runner))