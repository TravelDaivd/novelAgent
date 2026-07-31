import logging

import uvicorn
from fastapi import  HTTPException, FastAPI
from starlette.middleware.cors import CORSMiddleware

from ai_agent.api.schemas import ChatRequest, ChatResponse, ChapterUpload, UploadResponse
from ai_agent.agent_system import AgentSystem
from service.chapter_processor import ChapterProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ApiMain:
    
    agentSystem = AgentSystem()
    chapterProcessor = ChapterProcessor()
    
    
    app = FastAPI(
        title="小说智能分析系统",
        description="基于三个自训练模型的小说分析后端",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/api/v1/upload")
    async def upload(chapter:ChapterUpload):
        try:
            content = chapter.content
            title = chapter.title
            chroma_vector_data_list,segment_count,unique_chapters = ApiMain.chapterProcessor.process_text_context(content,title)
            entites_num = 0
            relation_num = 0
            for chroma in chroma_vector_data_list:
                entites_num  += len(chroma.get("entities"))
                if "relations" in chroma:
                    relation_num  += len(chroma.get("relations"))

            
            
            return UploadResponse(
                status= "success",
                chapter_id= unique_chapters,
                segment_count= segment_count,
                entity_count= entites_num,
                relation_count= relation_num
            )
            
            
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/chat")
    async def chat(request:ChatRequest):
        try:
            question = request.question
            answer =  ApiMain.agentSystem.execute_question(question)
            return ChatResponse(
                answer=answer,
                confidence=0.95
            )
        except Exception as e:
            logger.error(f"出现未知错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root(self):
        return {
            "message": "小说智能分析系统",
            "version": "1.0.0",
            "endpoints": {
                "chat": "POST /api/v1/chat"
            }
        }
if __name__ == "__main__":
    uvicorn.run(
        ApiMain.app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
    
    
    
    
    
    
    
    
    