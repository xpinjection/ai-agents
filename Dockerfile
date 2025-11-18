FROM langchain/langgraph-api:3.13

# Copy sources and install all dependencies
COPY src /deps/ai-agents/src
COPY langgraph.json pyproject.toml /deps/ai-agents/
RUN cd /deps/ai-agents && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .

# Configure graphs
ENV LANGSERVE_GRAPHS='{"chat": "/deps/ai-agents/src/simple-chat/simple_chat.py:simple_chat", "chef": "/deps/ai-agents/src/coooking/cooking.py:cooking_chef", "flights": "/deps/ai-agents/src/flights/flight_assistant.py:create_flight_assistant", "conventions": "/deps/ai-agents/src/conventions/conventions.py:conventions_assistant", "conventions_agentic": "/deps/ai-agents/src/conventions/conventions_agentic.py:conventions_agentic_assistant", "spending": "/deps/ai-agents/src/transactions/spending_assistant.py:create_spending_agent", "prompt_chaining": "/deps/ai-agents/src/workflows/prompt_chaining.py:prompt_chain_agent", "parallel": "/deps/ai-agents/src/workflows/parallelization.py:parallel_agent", "routing": "/deps/ai-agents/src/workflows/routing.py:routing_agent", "evaluator_optimizer": "/deps/ai-agents/src/workflows/evaluator_optimizer.py:evaluator_optimizer_agent", "supervisor": "/deps/ai-agents/src/workflows/supervisor.py:supervisor_assistant"}'

# Override installed dependencies with langgraph-api dependencies
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

# Removing build dependencies
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

# Setup workdir to copied project
WORKDIR /deps/ai-agents