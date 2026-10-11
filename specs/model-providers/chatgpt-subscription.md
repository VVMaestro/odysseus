# ChatGPT Subscription Provider Shape

Last updated: 2026-10-11

## Scope

Canonical provider ID `chatgpt_subscription`; Codex Responses transport;
auth and runtime code in `src/chatgpt_subscription.py`,
`routes/chatgpt_subscription_routes.py`, and `src/llm_core.py`.
There is no dedicated ChatGPT Subscription canonical reader on current `dev`.

## Catalog Shape

The account-scoped Codex models endpoint returns root `models[]`; `slug` is the
request identity and `visibility`/`priority` control availability/order. These
fields do not prove tools, reasoning, vision, or context. Null/malformed model
lists fail soft rather than crashing discovery (#5280/#5281).

The canonical generic reader does not accept `slug`-only items, so this runtime
catalog is not currently normalized into `ModelCapabilityRecord` values.

## Request And Response Shape

Transport uses a ChatGPT backend Responses endpoint, `input` items, flattened
function tools, streamed function-call argument events, exact `call_id`, and
`function_call_output` continuation. Parallel calls and encrypted reasoning
continuity require preserving typed output/history rather than coercing all
roles to text. The adapter sends selected built-in and MCP function schemas,
accumulates streamed argument fragments, and emits the agent's native
`tool_calls` event once before completion. Final item/response snapshots replace
arguments instead of appending duplicate calls. Assistant calls and tool results
are replayed in order as `function_call` and `function_call_output`, using the
original `call_id` (not the output item's `id`). Chat schemas without an explicit
strict setting use `strict: false` to retain their optional-argument semantics.

New Subscription endpoints enable native tools. A one-time startup migration
enables existing endpoints linked to ChatGPT Subscription auth on the Codex
backend URL; other providers and custom API URLs are untouched. Historic
`supports_tools=False` values cannot distinguish the old adapter default from a
user choice, because reconnect previously forced False. This migration resets
those values once. Subsequent explicit False values survive startup and reconnect
and retain the agent's text tool mode. NULL uses the Subscription native default.

The payload still omits `max_output_tokens` and uses `store: false`. Automated
transport tests use synthetic SSE; a real two-round Subscription call and a
separate MCP call remain manual acceptance checks.

OAuth/device credentials and refresh are provider-session behavior. Expired
credentials should return an actionable reconnect error, not generic model
failure.

## Fallback And Safety

Only the explicit internal base/ChatGPT host selects this provider. Never send
subscription credentials to a custom OpenAI-compatible URL. Catalog slugs stay
identity-only unless account-scoped fields or probes supply capability.

## Current Gaps

- Comprehensive Responses tool/reasoning parity is still evolving.
- Account model slugs are not consumed by the canonical reader package.
- The account catalog does not currently provide a complete canonical
  capability card for every slug.
