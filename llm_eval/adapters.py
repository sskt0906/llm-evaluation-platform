"""Model boundary. Exceptions carry only an allowlisted category."""
from dataclasses import dataclass
from typing import Protocol
import os


@dataclass
class Reply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Failure(Exception):
    pass


class Adapter(Protocol):
    def generate(self, prompt: str, timeout: float) -> Reply: ...


class Mock:
    def __init__(self, cases, degraded):
        self.cases = {c["prompt"]: c for c in cases}
        self.degraded = degraded

    def generate(self, prompt, timeout):
        case = self.cases[prompt]
        if case.get("fault"):
            raise Failure(case["fault"])
        output = case["bad" if self.degraded else "good"]
        return Reply(output, len(prompt.split()), len(output.split()))


class Live:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

    def generate(self, prompt, timeout):
        try:
            return self._generate(prompt, timeout)
        except Failure:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", getattr(exc, "code", None))
            if status in (429, 500):
                raise Failure(str(status)) from None
            if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
                raise Failure("timeout") from None
            raise Failure("provider_error") from None

    def _generate(self, prompt, timeout):
        if self.provider == "ollama":
            import json
            from urllib.request import Request, build_opener, ProxyHandler
            from urllib.error import HTTPError

            request = Request(
                "http://127.0.0.1:11434/api/generate",
                json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode(),
                {"Content-Type": "application/json"},
            )
            try:
                with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
                    result = json.load(response)
            except HTTPError as exc:
                raise Failure(str(exc.code) if exc.code in (429, 500) else "provider_error") from None
            return Reply(result.get("response"), result.get("prompt_eval_count"),
                         result.get("eval_count"))
        if self.provider == "openai":
            from openai import OpenAI

            with OpenAI(timeout=timeout, max_retries=0) as client:
                result = client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}])
            usage = result.usage
            return Reply(result.choices[0].message.content,
                         usage.prompt_tokens if usage else None,
                         usage.completion_tokens if usage else None)
        if self.provider == "anthropic":
            from anthropic import Anthropic

            with Anthropic(timeout=timeout, max_retries=0) as client:
                result = client.messages.create(model=self.model, max_tokens=1024,
                                                messages=[{"role": "user", "content": prompt}])
            return Reply("".join(block.text for block in result.content if block.type == "text"),
                         result.usage.input_tokens, result.usage.output_tokens)
        if self.provider == "gemini":
            from google import genai
            from google.genai import types

            options = types.HttpOptions(timeout=int(timeout * 1000),
                                        retry_options=types.HttpRetryOptions(attempts=1))
            with genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=options) as client:
                result = client.models.generate_content(model=self.model, contents=prompt)
            usage = result.usage_metadata
            return Reply(result.text, usage.prompt_token_count if usage else None,
                         usage.candidates_token_count if usage else None)
        raise Failure("provider_error")
