"""Chat completions API endpoints conforming to OpenAI-compatible specifications.

Coordinates PII sanitization, semantic caching via Redis vector similarity,
intent-based routing, circuit breaker resilience, and fallback execution.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeltaContent,
    ErrorResponse,
    UsageInfo,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AllProvidersFailedError,
    CircuitOpenError,
    ProviderAPIError,
    RateLimitError,
)
from app.core.logging import get_logger
from app.telemetry.metrics import (
    record_cache_operation,
    record_pii_detection,
    record_request,
    record_request_duration,
    record_tokens,
    record_ttft,
    update_circuit_state,
)

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])


def _extract_prompt_text(messages: list[ChatMessage]) -> str:
    """Extract user prompt text from conversation messages for caching and routing.

    Args:
        messages: List of chat messages in the conversation.

    Returns:
        str: Concatenated user message text, or full conversation text if no user messages exist.
    """
    user_contents = [m.content for m in messages if m.role == "user" and m.content]
    if user_contents:
        return "\n".join(user_contents)
    return "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)


def _estimate_token_count(text: str) -> int:
    """Rough heuristic estimate of token count (4 chars/token).

    Args:
        text: Input string.

    Returns:
        int: Estimated token count.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create Chat Completion",
    description="Generate chat completions with semantic caching, intent routing, and fallback resilience.",
)
@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    include_in_schema=False,
)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """Handle chat completion requests with caching, routing, resilience, and telemetry.

    Execution Flow:
    1. Resolve authenticated client identifier from request state.
    2. Sanitize request messages using regex PII redaction.
    3. Query Redis vector semantic cache; return immediately on hit.
    4. Dynamically route to optimal provider and model based on intent heuristics.
    5. Execute against upstream providers using circuit breaker and fallback chain.
    6. Record telemetry metrics (TTFT, token counts, latency, and status).
    7. Asynchronously store newly generated responses in semantic cache.

    Args:
        request: Unified chat completion request payload.
        raw_request: Underlying FastAPI request containing application state and headers.

    Returns:
        ChatCompletionResponse | StreamingResponse: Non-streaming completion response or SSE stream.

    Raises:
        AllProvidersFailedError: If all providers in the fallback chain fail or are unavailable.
    """
    start_time = time.perf_counter()

    # 1. Resolve client identity and application services
    client_id: str = (
        getattr(raw_request.state, "client_id", None)
        or raw_request.headers.get("x-client-id")
        or "anonymous"
    )
    settings: Settings = getattr(raw_request.app.state, "settings", None) or get_settings()
    sanitizer = getattr(raw_request.app.state, "sanitizer", None)
    semantic_cache = getattr(raw_request.app.state, "semantic_cache", None)
    intent_router = getattr(raw_request.app.state, "router", None)
    circuit_breakers = getattr(raw_request.app.state, "circuit_breakers", None)
    providers = getattr(raw_request.app.state, "providers", {})

    # 2. PII Sanitization
    if sanitizer is not None:
        try:
            msg_dicts = [m.model_dump() for m in request.messages]
            sanitized_dicts, pii_detected = sanitizer.redact_messages(msg_dicts)
            request.messages = [ChatMessage(**m) for m in sanitized_dicts]

            if pii_detected:
                logger.info(
                    "PII detected and redacted from request",
                    client_id=client_id,
                    pii_types=pii_detected,
                )
                for pii_type in pii_detected:
                    record_pii_detection(pii_type)
        except Exception as pii_err:
            logger.warning("PII sanitization encounter error", error=str(pii_err))

    # 3. Semantic Cache Lookup
    prompt_text = _extract_prompt_text(request.messages)
    cached_text: str | None = None

    if semantic_cache is not None and prompt_text:
        try:
            cached_text = await semantic_cache.lookup(prompt_text)
        except Exception as cache_err:
            logger.warning("Semantic cache lookup error", error=str(cache_err))
            record_cache_operation("lookup", "error")

    # 3a. Cache Hit Handling
    if cached_text is not None:
        record_cache_operation("lookup", "hit")
        target_model = request.model or "cached"
        prompt_tokens = sum(_estimate_token_count(m.content) for m in request.messages)
        completion_tokens = _estimate_token_count(cached_text)

        duration = time.perf_counter() - start_time
        record_request(
            provider="cache",
            model=target_model,
            status="200",
            cache_hit=True,
            client_id=client_id,
        )
        record_tokens(
            provider="cache",
            model=target_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        record_request_duration(
            provider="cache",
            model=target_model,
            duration=duration,
        )

        # Cache Hit - Streaming Response
        if request.stream:

            async def _stream_cached_response() -> AsyncGenerator[str, None]:
                completion_id = f"chatcmpl-cached-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())

                # Role and content delta chunk
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=target_model,
                    choices=[
                        Choice(
                            index=0,
                            delta=DeltaContent(role="assistant", content=cached_text),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Stop chunk
                stop_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=target_model,
                    choices=[
                        Choice(
                            index=0,
                            delta=DeltaContent(role=None, content=None),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {stop_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _stream_cached_response(),
                media_type="text/event-stream",
            )

        # Cache Hit - Non-Streaming Response
        return ChatCompletionResponse(
            id=f"chatcmpl-cached-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=cached_text),
                    finish_reason="stop",
                )
            ],
            model=target_model,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cached=True,
        )

    if semantic_cache is not None and prompt_text:
        record_cache_operation("lookup", "miss")

    # 4. Intent Routing
    headers_dict = dict(raw_request.headers)
    if intent_router is not None:
        routed_provider, routed_model = intent_router.route(request, headers_dict)
    else:
        routed_provider = request.provider or settings.DEFAULT_PROVIDER
        routed_model = request.model or "mock-gpt-4o"

    if not request.model:
        request.model = routed_model

    selected_provider = request.provider or routed_provider

    # 5. Build Fallback Chain
    fallback_chain: list[str] = [selected_provider]
    for p in settings.FALLBACK_CHAIN:
        if p not in fallback_chain:
            fallback_chain.append(p)

    # Always ensure default provider and mock are included as safety nets
    for safety_provider in (settings.DEFAULT_PROVIDER, "mock"):
        if safety_provider not in fallback_chain and safety_provider in providers:
            fallback_chain.append(safety_provider)

    logger.info(
        "Executing chat completion with fallback chain",
        selected_provider=selected_provider,
        fallback_chain=fallback_chain,
        model=request.model,
        stream=request.stream,
        client_id=client_id,
    )

    # 6. Streaming Mode Execution
    if request.stream:

        async def _stream_generator() -> AsyncGenerator[str, None]:
            nonlocal start_time
            active_stream: AsyncGenerator[ChatCompletionChunk, None] | None = None
            successful_provider: str | None = None
            accumulated_chunks: list[str] = []
            ttft_recorded = False

            # Try providers in fallback chain until stream is initialized
            for provider_name in fallback_chain:
                provider_instance = providers.get(provider_name)
                if provider_instance is None:
                    continue

                cb = circuit_breakers.get(provider_name) if circuit_breakers else None
                try:
                    if cb is not None and not await cb.can_execute():
                        raise CircuitOpenError(provider_name=provider_name)

                    # Initialize stream iterator
                    stream_iter = provider_instance.stream(request).__aiter__()

                    # Obtain the first chunk to guarantee provider is operational
                    timeout_sec = settings.CB_REQUEST_TIMEOUT_MS / 1000.0
                    try:
                        first_chunk = await asyncio.wait_for(
                            stream_iter.__anext__(),
                            timeout=timeout_sec,
                        )
                    except StopAsyncIteration:
                        first_chunk = None

                    successful_provider = provider_name
                    active_stream = stream_iter

                    # Record TTFT on first token chunk
                    ttft = time.perf_counter() - start_time
                    record_ttft(
                        provider=provider_name,
                        model=request.model or "unknown",
                        duration=ttft,
                    )
                    ttft_recorded = True

                    if cb is not None:
                        await cb._on_success()
                        update_circuit_state(provider_name, cb.state_value)

                    # Yield the first chunk
                    if first_chunk is not None:
                        if (
                            first_chunk.choices
                            and first_chunk.choices[0].delta
                            and first_chunk.choices[0].delta.content
                        ):
                            accumulated_chunks.append(first_chunk.choices[0].delta.content)
                        yield f"data: {first_chunk.model_dump_json()}\n\n"

                    break
                except Exception as stream_err:
                    logger.warning(
                        "Provider stream initialization failed, attempting next fallback",
                        provider=provider_name,
                        error=str(stream_err),
                    )
                    if cb is not None:
                        await cb._on_failure()
                        update_circuit_state(provider_name, cb.state_value)
                    continue

            # Check if all streaming attempts failed
            if successful_provider is None or active_stream is None:
                logger.error(
                    "All providers in streaming fallback chain failed",
                    fallback_chain=fallback_chain,
                )
                err_resp = ErrorResponse(
                    error="ALL_PROVIDERS_FAILED",
                    detail="All configured LLM providers in the fallback chain failed.",
                    code="ALL_PROVIDERS_FAILED",
                )
                yield f"data: {err_resp.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Consume the remaining stream chunks
            try:
                async for chunk in active_stream:
                    if not ttft_recorded:
                        ttft = time.perf_counter() - start_time
                        record_ttft(
                            provider=successful_provider,
                            model=request.model or "unknown",
                            duration=ttft,
                        )
                        ttft_recorded = True

                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        accumulated_chunks.append(chunk.choices[0].delta.content)

                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as stream_run_err:
                logger.error(
                    "Error during active token stream transmission",
                    provider=successful_provider,
                    error=str(stream_run_err),
                )
                cb = circuit_breakers.get(successful_provider) if circuit_breakers else None
                if cb is not None:
                    await cb._on_failure()
                    update_circuit_state(successful_provider, cb.state_value)
            finally:
                yield "data: [DONE]\n\n"

                # Store completed response in semantic cache
                full_stream_text = "".join(accumulated_chunks)
                duration = time.perf_counter() - start_time
                if semantic_cache is not None and full_stream_text and prompt_text:
                    try:
                        asyncio.create_task(semantic_cache.store(prompt_text, full_stream_text))
                        record_cache_operation("store", "success")
                    except Exception as store_err:
                        logger.warning(
                            "Failed to store streamed response in semantic cache",
                            error=str(store_err),
                        )
                        record_cache_operation("store", "error")

                p_toks = sum(_estimate_token_count(m.content) for m in request.messages)
                c_toks = _estimate_token_count(full_stream_text)
                record_request(
                    provider=successful_provider,
                    model=request.model or "unknown",
                    status="200",
                    cache_hit=False,
                    client_id=client_id,
                )
                record_tokens(
                    provider=successful_provider,
                    model=request.model or "unknown",
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                )
                record_request_duration(
                    provider=successful_provider,
                    model=request.model or "unknown",
                    duration=duration,
                )

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
        )

    # 7. Non-Streaming Execution
    response: ChatCompletionResponse | None = None
    successful_provider: str | None = None
    last_exception: Exception | None = None

    for provider_name in fallback_chain:
        provider_instance = providers.get(provider_name)
        if provider_instance is None:
            logger.warning("Provider not configured in proxy registry", provider=provider_name)
            continue

        cb = circuit_breakers.get(provider_name) if circuit_breakers else None
        try:
            if cb is not None:
                response = await cb.execute(provider_instance.generate, request)
                update_circuit_state(provider_name, cb.state_value)
            else:
                response = await provider_instance.generate(request)

            successful_provider = provider_name
            break
        except (CircuitOpenError, ProviderAPIError, RateLimitError, Exception) as exc:
            last_exception = exc
            logger.warning(
                "Provider invocation failed, falling back to next provider in chain",
                provider=provider_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if cb is not None:
                update_circuit_state(provider_name, cb.state_value)
            continue

    if response is None or successful_provider is None:
        logger.error(
            "All providers in fallback chain failed",
            fallback_chain=fallback_chain,
            last_error=str(last_exception),
        )
        raise AllProvidersFailedError(
            message=f"All configured LLM providers in fallback chain failed. Last error: {last_exception}",
            details={"fallback_chain": fallback_chain, "last_error": str(last_exception)},
        )

    # 8. Post-generation Cache Persistence & Telemetry Recording
    full_response_text = ""
    if response.choices and response.choices[0].message and response.choices[0].message.content:
        full_response_text = response.choices[0].message.content

    if semantic_cache is not None and full_response_text and prompt_text:
        try:
            asyncio.create_task(semantic_cache.store(prompt_text, full_response_text))
            record_cache_operation("store", "success")
        except Exception as store_err:
            logger.warning("Failed to persist response to semantic cache", error=str(store_err))
            record_cache_operation("store", "error")

    duration = time.perf_counter() - start_time
    prompt_tokens = (
        response.usage.prompt_tokens if response.usage else _estimate_token_count(prompt_text)
    )
    completion_tokens = (
        response.usage.completion_tokens
        if response.usage
        else _estimate_token_count(full_response_text)
    )

    record_request(
        provider=successful_provider,
        model=response.model or request.model or "unknown",
        status="200",
        cache_hit=False,
        client_id=client_id,
    )
    record_tokens(
        provider=successful_provider,
        model=response.model or request.model or "unknown",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    record_request_duration(
        provider=successful_provider,
        model=response.model or request.model or "unknown",
        duration=duration,
    )

    return response


__all__ = ["router", "chat_completions"]
