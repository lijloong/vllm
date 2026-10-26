# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD gateway billing plane.

`lwd_gateway_wire` holds the OpenAI-compatible wire shape shared by both
roles. `lwd_gateway_server` runs it on the cloud (SSE control endpoint),
`lwd_gateway_client` opens one stream per request on the edge so the platform
gateway can bill from the final `usage` frame.
"""
