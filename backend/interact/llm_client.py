import os
import time
import boto3
import botocore.config
import botocore.exceptions


class LLMClient:
    """
    Unified LLM caller.
    LLM_FLOW=server  → AWS Bedrock (Claude)
    LLM_FLOW=direct  → Anthropic / Gemini / OpenAI public APIs
    Falls back to TRANSLATION_FLOW if LLM_FLOW is not set.
    """

    def __init__(self):
        self.flow = os.getenv("LLM_FLOW", os.getenv("TRANSLATION_FLOW", "direct"))
        self.max_tokens = 8096
        self.max_retries = 3

        if self.flow == "server":
            region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
            self.model_id = os.getenv("BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
            self.bedrock = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=botocore.config.Config(
                    read_timeout=300,
                    connect_timeout=10,
                    retries={"max_attempts": 0},
                ),
            )
            print(f"[LLMClient] server flow → Bedrock model={self.model_id}, region={region}")
        else:
            print("[LLMClient] direct flow")

    def call(self, system: str, user: str, provider: str = "anthropic") -> str:
        """Call LLM and return text response."""
        if self.flow == "server":
            print(f"[LLM] call → BEDROCK | model={self.model_id}")
            return self._call_bedrock(system, user)
        model_hint = {
            "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            "gemini":    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "openai":    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }.get(provider, provider)
        print(f"[LLM] call → DIRECT | provider={provider} | model={model_hint}")
        return self._call_direct(system, user, provider)

    def stream(self, system: str, user: str, provider: str = "anthropic"):
        """Yield text tokens as a generator for streaming responses."""
        if self.flow == "server":
            print(f"[LLM] stream → BEDROCK | model={self.model_id}")
            yield from self._stream_bedrock(system, user)
        else:
            model_hint = {
                "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                "gemini":    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                "openai":    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            }.get(provider, provider)
            print(f"[LLM] stream → DIRECT | provider={provider} | model={model_hint}")
            yield from self._stream_direct(system, user, provider)

    def _stream_bedrock(self, system: str, user: str):
        resp = self.bedrock.converse_stream(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        for event in resp["stream"]:
            chunk = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if chunk:
                yield chunk

    def _stream_direct(self, system: str, user: str, provider: str):
        if provider == "gemini":
            yield from self._stream_gemini(system, user)
        elif provider == "openai":
            yield from self._stream_openai(system, user)
        elif provider == "anthropic":
            yield from self._stream_anthropic(system, user)
        else:
            # Unknown provider — cascade through available keys: gemini → openai → anthropic
            if os.getenv("GEMINI_API_KEY"):
                yield from self._stream_gemini(system, user)
            elif os.getenv("OPENAI_API_KEY"):
                yield from self._stream_openai(system, user)
            else:
                yield from self._stream_anthropic(system, user)

    def _stream_anthropic(self, system: str, user: str):
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        with client.messages.stream(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_gemini(self, system: str, user: str):
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=user,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_tokens,
                temperature=0.0,
            ),
        ):
            if chunk.text:
                yield chunk.text

    def _stream_openai(self, system: str, user: str):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        for chunk in client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
            stream=True,
        ):
            text = chunk.choices[0].delta.content
            if text:
                yield text

    # ------------------------------------------------------------------
    def _call_bedrock(self, system: str, user: str) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.bedrock.converse(
                    modelId=self.model_id,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
                )
                return resp["output"]["message"]["content"][0]["text"]
            except botocore.exceptions.ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("ThrottlingException", "ServiceUnavailableException") and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Bedrock API failed after retries")

    def _call_direct(self, system: str, user: str, provider: str) -> str:
        if provider == "gemini":
            return self._call_gemini(system, user)
        if provider == "openai":
            return self._call_openai(system, user)
        if provider == "anthropic":
            return self._call_anthropic(system, user)
        # Unknown provider — cascade through available keys: gemini → openai → anthropic
        if os.getenv("GEMINI_API_KEY"):
            return self._call_gemini(system, user)
        if os.getenv("OPENAI_API_KEY"):
            return self._call_openai(system, user)
        return self._call_anthropic(system, user)

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        msg = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    def _call_gemini(self, system: str, user: str) -> str:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_tokens,
                temperature=0.0,
            ),
        )
        return response.text

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content
