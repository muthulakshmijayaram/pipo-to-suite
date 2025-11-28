import asyncio
import sys
import os
import json
import logging
import re
import threading
import time
import signal
from typing import Optional, Any, Type, List
from contextlib import AsyncExitStack

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PModel

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    filename="pipo_client.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# =====================================================================
# SAP GEN AI HUB LLM
# =====================================================================
def create_sap_llm() -> ChatOpenAI:
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )


# =====================================================================
# ID HELPERS
# =====================================================================
def make_iflow_id_from_query(query: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")
    if not slug:
        slug = "GenericIFlow"
    if len(slug) > 40:
        slug = slug[:40]
    if slug[0].isdigit():
        slug = f"IF_{slug}"
    return slug


def make_package_id_from_iflow_id(iflow_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]", "", iflow_id)
    if not base:
        base = "GenericIFlow"
    if base[0].isdigit():
        base = "P" + base
    if not base.endswith("Pkg"):
        base += "Pkg"
    return base


def make_package_name_from_iflow_id(iflow_id: str) -> str:
    return f"{iflow_id} Package"


# =====================================================================
# BPMN GENERATOR
# =====================================================================
class IntelligentBPMNGenerator:
    def __init__(self, llm: ChatOpenAI, reference_bpmn_path: str):
        self.llm = llm
        self.reference_bpmn_path = reference_bpmn_path
        self.reference_bpmn = None

    def load_reference_bpmn(self) -> str:
        if self.reference_bpmn:
            return self.reference_bpmn

        if not os.path.exists(self.reference_bpmn_path):
            raise FileNotFoundError(f"Reference BPMN not found: {self.reference_bpmn_path}")

        with open(self.reference_bpmn_path, "r", encoding="utf-8") as f:
            self.reference_bpmn = f.read()

        logger.info(f"[BPMN-REF] Loaded reference from {self.reference_bpmn_path}")
        return self.reference_bpmn

    async def generate_bpmn(self, requirement: str, iflow_id: str) -> str:
        reference = self.load_reference_bpmn()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SAP Cloud Integration BPMN generator..."),
            ("human", "User Requirement: {requirement}\n\nREFERENCE:\n{reference}")
        ])

        response = await prompt.ainvoke({
            "requirement": requirement,
            "reference": reference
        })

        result = await self.llm.ainvoke(response.messages)
        xml = result.content.strip()

        if "```" in xml:
            xml = xml.split("```")[1].split("```")[0].strip()

        if not xml.startswith("<?xml"):
            xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml

        return xml


# =====================================================================
# MCP TOOL WRAPPER
# =====================================================================
class MCPAsyncTool(BaseTool):
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str

    async def _arun(self, *args, **kwargs) -> str:
        result = await self.session.call_tool(self.mcp_tool_name, kwargs)
        if not result.content:
            return ""

        outputs = []
        for c in result.content:
            if getattr(c, "text", None):
                outputs.append(c.text)
            elif getattr(c, "json", None):
                outputs.append(json.dumps(c.json))
            else:
                outputs.append(str(c))

        return "\n".join(outputs)

    def _run(self, *args, **kwargs) -> str:
        return asyncio.run(self._arun(*args, **kwargs))


# =====================================================================
# MCP CLIENT
# =====================================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.llm = create_sap_llm()
        self.agent_tools: List[BaseTool] = []
        self.worker_agent: Optional[AgentExecutor] = None
        self.bpmn_generator: Optional[IntelligentBPMNGenerator] = None

    async def connect_to_server(self, script: str):
        cmd = "python" if script.endswith(".py") else "node"
        params = StdioServerParameters(command=cmd, args=[script])

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        await self._load_tools()

        reference_path = os.getenv("CPI_DEMO_IFLW_PATH", "demo.iflw")
        self.bpmn_generator = IntelligentBPMNGenerator(self.llm, reference_path)

    async def _load_tools(self):
        tool_defs = await self.session.list_tools()
        for tdef in tool_defs.tools:
            schema = tdef.inputSchema or {}
            props = schema.get("properties", {})
            req = schema.get("required", [])

            fields = {k: (Any, ... if k in req else None) for k in props}
            InputModel = create_model(f"{tdef.name}_Input", **fields)

            tool = MCPAsyncTool(
                name=tdef.name, description=tdef.description,
                args_schema=InputModel, session=self.session,
                mcp_tool_name=tdef.name
            )
            self.agent_tools.append(tool)

    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
        except Exception:
            pass

    async def process_query(self, text: str):
        if not self.bpmn_generator:
            return "BPMN Generator not initialized"

        iflow_id = make_iflow_id_from_query(text)
        return await self.bpmn_generator.generate_bpmn(text, iflow_id)


# =====================================================================
# ============ FASTAPI SERVER ROUTES =================================
# =====================================================================

app = FastAPI(title="MCP Integration Suite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mcp_client_api = MCPClient()
is_connected = False

class ChatRequest(PModel):
    message: str

class ConnectRequest(PModel):
    server_path: str


# ---------------------------------------------------------------------
# 1️⃣ CONNECT ROUTE  (OK)
# ---------------------------------------------------------------------
@app.post("/connect")
async def connect_api(req: ConnectRequest):
    global is_connected

    if is_connected:
        return {"status": "already_connected"}

    try:
        await mcp_client_api.connect_to_server(req.server_path)
        is_connected = True
        return {"status": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 2️⃣ CHAT ROUTE — ⭐⭐⭐ UPDATED ⭐⭐⭐
# ---------------------------------------------------------------------
@app.post("/chat")
async def chat_api(req: ChatRequest):
    if not is_connected:
        raise HTTPException(400, "Not connected")

    try:
        response = await mcp_client_api.process_query(req.message)

        # ⭐⭐⭐ PRINT RESPONSE TO TERMINAL ⭐⭐⭐
        print("\n================ CHAT RESPONSE ================")
        print(response)
        print("===============================================\n")

        return {"response": response}

    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------
# 3️⃣ DISCONNECT ROUTE — ⭐⭐⭐ UPDATED ⭐⭐⭐
# ---------------------------------------------------------------------

def kill_server():
    """Fully stop the FastAPI server process."""
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)


@app.post("/disconnect")
async def disconnect_api():
    global is_connected

    try:
        await mcp_client_api.cleanup()
        is_connected = False

        print("\n🔥 Server disconnecting & shutting down…\n")

        # ⭐⭐⭐ EXIT SERVER COMPLETELY ⭐⭐⭐
        threading.Thread(target=kill_server, daemon=True).start()

        return {"status": "disconnected_and_exited"}

    except Exception as e:
        raise HTTPException(500, str(e))


# =====================================================================
# RUN MODE
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
    else:
        print("Run with: python main.py api")
