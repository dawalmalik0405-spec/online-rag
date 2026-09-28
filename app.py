import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Annotated, List

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.chat_message_histories import FileChatMessageHistory
from pydantic import BaseModel

from llm_engine import llm_core, llm_stream
from vectordb import delete_document_collection, ingest_file


app = FastAPI(title="RAG AI Assistant")

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}

history = {}
os.makedirs("history", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

uploads = os.path.join(os.getcwd(), "uploads")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/save/")
async def file_save(request: Request, file: Annotated[UploadFile, File()]):
    clean_filename = Path(file.filename).name
    extension = Path(clean_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
        )

    file_location = os.path.join(uploads, clean_filename)
    try:
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)

        # Offload heavy CPU/embeddings ingestion to threadpool
        result = await asyncio.to_thread(ingest_file, file_location, clean_filename)
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to ingest document: {str(e)}"}
        )

    return JSONResponse({
        "status": "success",
        "filename": clean_filename,
        "chunks": result.get("stored_chunks", 0),
        "message": f"File '{clean_filename}' uploaded and indexed successfully!"
    })


class SourceInfo(BaseModel):
    chunk_index: int
    similarity: float
    snippet: str


class ChatRequest(BaseModel):
    message: str
    filename: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.filename:
        return ChatResponse(answer="Please select a document first.", sources=[])

    clean_filename = Path(request.filename).name

    if clean_filename not in history:
        history_path = os.path.join("history", f"{clean_filename}_history.json")
        history[clean_filename] = FileChatMessageHistory(file_path=history_path)

    chat_history = history[clean_filename]

    query = request.message.strip()
    if not query:
        return ChatResponse(answer="Please enter a question.", sources=[])

    chat_history.add_user_message(query)
    formatted_history = "\n".join(
        f"{msg.type.upper()}: {msg.content}"
        for msg in chat_history.messages
    )

    # Offload LLM and retrieval call to threadpool
    result = await asyncio.to_thread(
        llm_core,
        query,
        formatted_history,
        clean_filename
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    chat_history.add_ai_message(answer)

    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.filename:
        async def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': 'Please select a document first.'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    clean_filename = Path(request.filename).name

    if clean_filename not in history:
        history_path = os.path.join("history", f"{clean_filename}_history.json")
        history[clean_filename] = FileChatMessageHistory(file_path=history_path)

    chat_history = history[clean_filename]

    query = request.message.strip()
    if not query:
        async def empty_gen():
            yield f"data: {json.dumps({'type': 'error', 'data': 'Please enter a question.'})}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    chat_history.add_user_message(query)
    formatted_history = "\n".join(
        f"{msg.type.upper()}: {msg.content}"
        for msg in chat_history.messages
    )

    async def event_generator():
        # Iterate over streaming generator in a background thread
        queue = asyncio.Queue()

        def producer():
            try:
                for event in llm_stream(query, formatted_history, clean_filename):
                    queue.put_nowait(event)
            except Exception as e:
                queue.put_nowait({"type": "error", "data": str(e)})
            finally:
                queue.put_nowait(None)  # Sentinel to signal completion

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, producer)

        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "done":
                full_text = event.get("full_answer", "")
                if full_text:
                    chat_history.add_ai_message(full_text)
            yield f"data: {json.dumps(event)}\n\n"

        await future

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@app.get("/files")
def list_uploaded_files():
    files = []
    if os.path.exists("uploads"):
        for f in os.listdir("uploads"):
            if os.path.isfile(os.path.join("uploads", f)):
                files.append(f)
    return {"files": sorted(files)}


@app.delete("/files/{filename}")
async def delete_file_endpoint(filename: str):
    clean_filename = Path(filename).name
    file_location = os.path.join(uploads, clean_filename)

    # 1. Remove physical file
    if os.path.exists(file_location):
        os.remove(file_location)

    # 2. Delete ChromaDB isolated collection
    delete_document_collection(clean_filename)

    # 3. Clean history
    history_file_path = os.path.join("history", f"{clean_filename}_history.json")
    if os.path.exists(history_file_path):
        os.remove(history_file_path)
    history.pop(clean_filename, None)

    return {"status": "success", "message": f"Document '{clean_filename}' deleted."}


@app.get("/history")
async def history_file(filename: str | None = None):
    if not filename:
        return {"messages": []}

    clean_filename = Path(filename).name

    if clean_filename not in history:
        history_path = os.path.join("history", f"{clean_filename}_history.json")
        if os.path.exists(history_path):
            history[clean_filename] = FileChatMessageHistory(file_path=history_path)
        else:
            return {"messages": []}

    chat_history = history[clean_filename]

    messages = [
        {
            "role": msg.type,
            "content": msg.content
        }
        for msg in chat_history.messages
    ]

    return {"messages": messages}




