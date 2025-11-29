 
import asyncio
import sys
import os
import json
import logging
from typing import Optional, Any, Type, Union, Dict, List
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import create_model, BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
import re
from yaspin import yaspin
 
 
# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    filename="pipo_client.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
 
load_dotenv()
 
 
# ==========================================================
# SAP GEN AI HUB LLM
# ==========================================================
def create_sap_llm():
    deployment_id = os.getenv("LLM_DEPLOYMENT_ID")
    if not deployment_id:
        raise RuntimeError("LLM_DEPLOYMENT_ID missing in .env")
 
    return ChatOpenAI(
        deployment_id=deployment_id,
        temperature=0,
    )
 
 
def build_pydantic_model(name: str, schema: Dict, root: Dict = None) -> Any:
    """
    Recursively converts MCP JSON schema into a Pydantic model.
    Supports objects, arrays, enums, oneOf, anyOf, $ref.
    """
    if root is None:
        root = schema
 
    # $ref support
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return Any
        path = ref[2:].split("/")
        target = root
        for part in path:
            target = target.get(part, {})
        return build_pydantic_model(name, target, root)
 
    # enums
    if "enum" in schema:
        from typing import Literal
        values = tuple(schema["enum"])
        return Literal[values]
 
    # oneOf / anyOf
    if "oneOf" in schema:
        subs = [build_pydantic_model(f"{name}_oneOf_{i}", s, root)
                for i, s in enumerate(schema["oneOf"])]
        return Union[tuple(subs)]
 
    if "anyOf" in schema:
        subs = [build_pydantic_model(f"{name}_anyOf_{i}", s, root)
                for i, s in enumerate(schema["anyOf"])]
        return Union[tuple(subs)]
 
    # Object
    if schema.get("type") == "object":
        props = schema.get("properties", {}) or {}
        required = schema.get("required", [])
 
        fields = {}
        for key, subschema in props.items():
            field_type = build_pydantic_model(f"{name}_{key}", subschema, root)
            default = ... if key in required else None
            fields[key] = (field_type, default)
 
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return create_model(safe_name, **fields)
 
    # Array
    if schema.get("type") == "array":
        item_schema = schema.get("items", {}) or {}
        item_type = build_pydantic_model(f"{name}_item", item_schema, root)
        return List[item_type]
 
    # Primitive
    if schema.get("type") == "string":
        return str
    if schema.get("type") == "integer":
        return int
    if schema.get("type") == "number":
        return float
    if schema.get("type") == "boolean":
        return bool
 
    # Fallback
    return Any
 
 
# ==========================================================
# ASYNC MCP TOOL WRAPPER
# ==========================================================
class MCPAsyncTool(BaseTool):
    """
    LangChain tool that calls an MCP tool asynchronously.
    Enhanced with better error handling and result formatting.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: ClientSession
    mcp_tool_name: str
 
    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Sync run is not supported; use async.")
 
    async def _arun(self, *args, **kwargs) -> str:
        logger.info(f"[MCP-TOOL] Executing → {self.mcp_tool_name} | Args = {kwargs}")
 
        try:
            with yaspin(text=f"Running MCP tool: {self.mcp_tool_name}", color="magenta") as sp:
                result = await self.session.call_tool(self.mcp_tool_name, kwargs)
                sp.ok("✔")
        except Exception as e:
            error_msg = f"ERROR: Tool {self.mcp_tool_name} failed with: {str(e)}"
            logger.error(f"[MCP-TOOL] {error_msg}")
            return error_msg
 
        if not result.content:
            warning = f"WARNING: Tool {self.mcp_tool_name} returned empty content"
            logger.warning(f"[MCP-TOOL] {warning}")
            return warning
 
        outputs = []
        has_error = False
       
        for c in result.content:
            if getattr(c, "text", None):
                text_content = c.text
                if "error" in text_content.lower() or "failed" in text_content.lower():
                    has_error = True
                outputs.append(text_content)
            elif getattr(c, "json", None):
                json_content = json.dumps(c.json, ensure_ascii=False, indent=2)
                outputs.append(json_content)
            elif hasattr(c, "error"):
                has_error = True
                outputs.append(f"ERROR: {c.error}")
            else:
                outputs.append(str(c))
 
        final_output = "\n".join(outputs)
       
        if has_error:
            logger.error(f"[MCP-TOOL] {self.mcp_tool_name} returned errors: {final_output}")
        else:
            logger.info(f"[MCP-TOOL] {self.mcp_tool_name} completed successfully")
 
        return final_output
 
 
# ==========================================================
# MCP CLIENT
# ==========================================================
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.llm = create_sap_llm()
        self.agent_tools: list[BaseTool] = []
        self.worker_agent: Optional[AgentExecutor] = None
 
    async def connect_to_server(self, server_script: str):
        """Connect to MCP server and initialize tools"""
        if server_script.endswith(".py"):
            cmd = "python"
        elif server_script.endswith(".js"):
            cmd = "node"
        else:
            raise ValueError("Server script must be .py or .js")
 
        params = StdioServerParameters(command=cmd, args=[server_script])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport
 
        with yaspin(text="Connecting to MCP server...", color="cyan") as sp:
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            sp.ok("✔")
       
        await self.session.initialize()
        await self._build_agent_tools()
        await self._build_worker_agent()
 
        tools = [t.name for t in (await self.session.list_tools()).tools]
        print("\n✅ MCP Connected! Available tools:", ", ".join(tools[:10]), "..." if len(tools) > 10 else "")
        logger.info(f"[MCP] Connected with {len(tools)} tools")
 
    async def _build_agent_tools(self):
        """Build LangChain tools from MCP tool definitions"""
        assert self.session is not None
        tool_list = await self.session.list_tools()
 
        for tdef in tool_list.tools:
            schema = tdef.inputSchema or {}
            InputModel = build_pydantic_model(f"{tdef.name}_Input", schema)
 
            tool = MCPAsyncTool(
                name=tdef.name,
                description=tdef.description or "",
                args_schema=InputModel,
                session=self.session,
                mcp_tool_name=tdef.name,
            )
            self.agent_tools.append(tool)
 
        logger.info(f"[MCP] Loaded {len(self.agent_tools)} tools")
 
    async def _build_worker_agent(self):
        """Build the worker agent with enhanced system prompt"""
        worker_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
 
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.agent_tools,
            prompt=worker_prompt,
        )
 
        self.worker_agent = AgentExecutor(
            agent=agent,
            tools=self.agent_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=25,
            early_stopping_method="force",
        )
 
    def _get_system_prompt(self) -> str:
        """Returns the enhanced system prompt with validation steps"""
        return """
You are a specialized assistant for SAP Integration Suite with focus on Integration Flows (IFlows).
 
## CRITICAL IFlow Creation and Update Guidelines
 
### IFlow File Structure (MANDATORY)
When creating or updating IFlows, the folder structure MUST be:
```
src/main/resources/
├── scenarioflows/
│   └── integrationflow/
│       └── <iflow_id>.iflw          # Main IFlow definition
├── mapping/                          # Message mappings
├── xsd/                              # XSD schema files
├── scripts/                          # Groovy/JavaScript scripts
└── parameters.prop                   # Optional parameters
```
 
### IFlow Creation Process (STEP-BY-STEP)
 
**Step 1: Check Prerequisites**
- First, check if package exists using `packages` or `package` tool
- If package doesn't exist, create it with `create-package`
- Verify package creation was successful
 
**Step 2: Create Empty IFlow**
- Use `create-empty-iflow` with:
  - package_name: The package technical name
  - iflow_id: Unique identifier (e.g., "if_json_to_xml")
  - iflow_name: Display name (e.g., "JSON to XML Converter")
  - iflow_description: Brief description
- WAIT for confirmation before proceeding
 
**Step 3: Verify IFlow Creation**
- IMMEDIATELY call `get-iflow` with the iflow_id
- Check that the response contains valid structure
- If error occurs here, the IFlow was not created properly - STOP and report
 
**Step 4: Update IFlow Content**
- Use `update-iflow` to add/modify files:
  - For .iflw file: path = "src/main/resources/scenarioflows/integrationflow/<iflow_id>.iflw"
  - For mappings: path = "src/main/resources/mapping/<mapping_name>.mmap"
  - For XSD: path = "src/main/resources/xsd/<schema_name>.xsd"
- Ensure XML content is properly formatted
- Include all required elements (Start, End, channels, processors)
 
**Step 5: Validate After Update**
- Call `get-iflow` again to verify changes were applied
- Check for any error indicators
- If you see "Error while loading details", the XML structure is invalid
 
**Step 6: Deploy**
- Only deploy after successful validation
- Use `deploy-iflow` with the iflow_id
- Check deployment status with `get-deploy-error` if deployment fails
 
### Common IFlow Components
 
**Sender Adapters** (Start of IFlow):
- HTTPS: For REST/HTTP endpoints
- SFTP: For file-based integration
- Mail: For email integration
- AMQP: For message queues
 
**Message Processors**:
- Content Modifier: Modify headers/body
- JSON to XML Converter: Transform JSON to XML
- XML to JSON Converter: Transform XML to JSON
- Groovy Script: Custom logic
- Message Mapping: Complex transformations
 
**Receiver Adapters** (End of IFlow):
- HTTP: Call external REST APIs
- SFTP: Write files
- Mail: Send emails
- SOAP: Call SOAP services
 
### Error Handling Protocol
 
**If ANY tool returns an error:**
1. Log the exact error message
2. Call `get-deploy-error` if it's a deployment error
3. Call `get-iflow` to check current state
4. Explain to user what went wrong
5. Suggest specific fixes
 
**Common Errors and Fixes:**
 
1. "Error while loading the details of the integration flow"
   - Cause: Malformed .iflw XML structure
   - Fix: Recreate the .iflw file with valid XML
   - Validate all XML tags are properly closed
 
2. "IFlow not found"
   - Cause: IFlow was not created or wrong ID used
   - Fix: Verify package and IFlow ID, create if missing
 
3. "Deployment failed"
   - Cause: Various (invalid adapter config, missing resources, etc.)
   - Fix: Use `get-deploy-error` to get specific error details
 
### Example Workflows
 
**Example 1: Create Simple HTTP to SFTP IFlow**
```
1. Check package exists → `packages`
2. Create IFlow → `create-empty-iflow`
3. Verify → `get-iflow`
4. Update with .iflw content → `update-iflow`
5. Verify again → `get-iflow`
6. Deploy → `deploy-iflow`
```
 
**Example 2: Add JSON to XML Converter**
```
1. Get existing IFlow → `get-iflow`
2. Update .iflw with JSON2XML processor → `update-iflow`
3. Verify changes → `get-iflow`
4. Test with message → `send-http-message`
5. Check results → `get-messages`
```
 
### VALIDATION CHECKLIST
 
Before marking any IFlow operation as complete, verify:
- ✓ IFlow exists (`get-iflow` returns valid data)
- ✓ All files are in correct paths
- ✓ XML structure is valid (no parsing errors)
- ✓ Required adapters are configured
- ✓ Deployment succeeds (if deploying)
- ✓ No errors in `get-deploy-error` or `get-messages`
 
### TPM Integration
 
For B2B scenarios with Trading Partner Management:
- Create Trading Partners first
- Set up Systems and Identifiers
- Create MIGs (Message Implementation Guidelines)
- Create Agreements
- Link IFlows to agreements
 
## Final Instructions
 
At the end of EVERY task, provide:
1. Summary of all tools executed (in order)
2. What each tool returned
3. Success/failure status
4. Any errors encountered with root cause
5. Next steps for user
6. Validation checklist status
 
REMEMBER: Always verify after create/update operations. Never assume success.
"""
 
    async def process_query(self, query: str) -> str:
        """Process user query with enhanced error tracking"""
        logger.info(f"[LLM] User Query: {query}")
 
        if not self.worker_agent:
            raise RuntimeError("Worker agent not initialized")
 
        intermediate_steps = []
       
        def log_agent_step(step):
            if isinstance(step, dict):
                intermediate_steps.append(step)
                if "tool" in step:
                    tool_name = step.get("tool", "unknown")
                    tool_input = step.get("tool_input", {})
                    logger.info(f"[LLM] Selected Tool → {tool_name} | Args → {tool_input}")
            return step
 
        try:
            with yaspin(text="Processing query...", color="yellow") as sp:
                worker_out = await self.worker_agent.ainvoke(
                    {"input": query},
                    callbacks=[log_agent_step],
                )
                sp.ok("✔")
        except Exception as e:
            logger.error(f"[LLM] Agent execution failed: {str(e)}")
            error_context = {
                "error": str(e),
                "steps_completed": len(intermediate_steps),
                "last_steps": intermediate_steps[-3:] if intermediate_steps else []
            }
            return f"❌ Error during execution:\n{str(e)}\n\nContext:\n{json.dumps(error_context, indent=2)}"
 
        raw_answer = worker_out.get("output", worker_out)
 
        # Analyze if there were errors
        has_errors = any([
            "ERROR" in raw_answer,
            "failed" in raw_answer.lower(),
            "error while loading" in raw_answer.lower(),
            "could not" in raw_answer.lower(),
        ])
 
        # Generate comprehensive summary
        summary_prompt = f"""
Analyze the following MCP tool execution results and provide a clear, actionable summary.
 
Tool Execution Results:
{raw_answer}
 
Intermediate Steps:
{json.dumps(intermediate_steps[-5:], indent=2) if intermediate_steps else "No intermediate steps recorded"}
 
Errors Detected: {"YES - Focus on error analysis" if has_errors else "NO - Operation appears successful"}
 
Provide a summary that includes:
 
1. **Operations Performed**: List each tool that was executed and what it did
2. **Results**: What did each tool return? Was it successful?
3. **Validation Status**: Were the changes verified? Did get-iflow confirm success?
4. **Errors (if any)**:
   - Exact error messages
   - Which component failed
   - Root cause analysis
   - Why it failed
5. **Next Steps**:
   - If successful: What the user can do now
   - If failed: Specific steps to fix the issue
6. **Recommendations**: Best practices or warnings
 
Format your response clearly with sections. Be specific and actionable.
"""
 
        with yaspin(text="Generating summary...", color="cyan") as sp:
            summary = await self.llm.apredict(summary_prompt)
            sp.ok("✔")
 
        logger.info(f"[LLM] Final Summary Generated")
       
        # Add visual indicator for errors
        if has_errors:
            summary = "⚠️ **ERRORS DETECTED**\n\n" + summary
        else:
            summary = "✅ **OPERATION COMPLETED**\n\n" + summary
 
        return summary
 
    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("SAP Integration Suite MCP Chatbot")
        print("="*60)
        print("Type your queries or 'quit' to exit\n")
 
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n💬 Query: ")
                if user_input.lower().strip() in ("quit", "exit", "q"):
                    print("\n👋 Goodbye!")
                    break
 
                if not user_input.strip():
                    continue
 
                answer = await self.process_query(user_input)
                print("\n" + "─"*60)
                print(answer)
                print("─"*60)
               
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                logger.exception("Unexpected error in chat loop")
 
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
 
 
# ==========================================================
# MAIN
# ==========================================================
async def main():
    if len(sys.argv) < 2:
        print("Usage: python pipo_client.py <server.js|server.py>")
        sys.exit(1)
 
    server = sys.argv[1]
    client = MCPClient()
   
    try:
        await client.connect_to_server(server)
        await client.chat_loop()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error in main")
    finally:
        await client.cleanup()
        
import os
import sys
import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading

# ==================================
# Your MCP Client must be defined above this line
# class MCPClient: ....
# It must contain:
#    - connect_to_server(path)
#    - process_query(query)
#    - exit_stack
# ==================================

app = FastAPI(title="MCP Integration API")

mcp_client = MCPClient()
is_connected = False


# ---------- REQUEST MODELS ----------

class QueryRequest(BaseModel):
    query: str
from pydantic import BaseModel

class DisconnectRequest(BaseModel):
    command: str



# ====================================
#               CONNECT
# ====================================



# ====================================
#                QUERY
@app.post("/query")
async def query_api(req: QueryRequest):
    global is_connected

    PATH = r"C:\\Users\Muthulakshmi Jayaram\\Documents\\mcp-integration-suite\\dist\\index.js"

    # Auto-connect if not connected
    if not is_connected:
        try:
            await mcp_client.connect_to_server(PATH)
            is_connected = True
            print(f"✅ MCP server auto-connected to: {PATH}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect: {str(e)}")

    try:
        # Call your MCP query processor
        result = await mcp_client.process_query(req.query)

        # Print to terminal also
        print("\n=========== QUERY ===========")
        print("Server Path :", PATH)
        print("Query       :", req.query)
        print("=========== RESULT ==========")
        print(result)
        print("=============================\n")

        return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_api(req: DisconnectRequest):
    command = req.command.lower()
    ...
    global is_connected

    if not is_connected:
        return {"status": "not connected"}

    # User must send quit/exit
    if command.lower() not in ["quit", "exit"]:
        return {"status": "invalid command", "message": "Send 'quit' or 'exit' to stop server"}

    try:
        # 🔌 Close MCP client
        if mcp_client.exit_stack:
            await mcp_client.exit_stack.aclose()

        is_connected = False

        # Return response first
        response = {"status": "disconnected", "message": "Server will exit now"}

        # 🔥 Stop FastAPI AFTER sending response
        def shutdown():
            time.sleep(0.2)
            sys.exit(0)

        threading.Thread(target=shutdown).start()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ====================================
#            MAIN ENTRY
# ====================================
if __name__ == "__main__":
    if "api" in sys.argv:
        uvicorn.run("main:app", host="0.0.0.0", port=8000)
    else:
        print("Run with: python main.py api")
    