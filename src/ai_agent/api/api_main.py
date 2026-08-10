import logging

import uvicorn
from fastapi import  HTTPException, FastAPI
from starlette.middleware.cors import CORSMiddleware

from ai_agent.agent_system import AgentSystem
from src.ai_agent.api.schemas import ChatRequest, ChatResponse, ChapterUpload, UploadResponse
from src.service.chapter_processor import ChapterProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ApiMain:
    
    def __init__(self):
        
        self.agentSystem = AgentSystem()
        self.chapterProcessor = ChapterProcessor()
        
        
        self.app = FastAPI(
            title="小说智能分析系统",
            description="基于三个自训练模型的小说分析后端",
            version="1.0.0"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # 注册路由
        self._setup_routes()
    
    def _setup_routes(self):
        """注册所有路由"""

        @self.app.post("/api/v1/upload")
        async def upload(chapter: ChapterUpload):
            return await self.upload(chapter)

        @self.app.post("/api/v1/chat")
        async def chat(request: ChatRequest):
            return await self.chat(request)

        @self.app.get("/")
        async def root():
            return {
                "message": "小说智能分析系统",
                "version": "1.0.0",
                "endpoints": {
                    "upload": "POST /api/v1/upload",
                    "chat": "POST /api/v1/chat"
                }
            }
    
    
    async def upload(self,chapter:ChapterUpload):
        try:
            content = chapter.content
            title = chapter.title
            chroma_vector_data_list,segment_count,unique_chapters = self.chapterProcessor.process_text_context(content,title)
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

    async def chat(self,request:ChatRequest):
        try:
            question = request.question
            answer =  self.agentSystem.execute_question(question)
            return ChatResponse(
                answer=answer,
                confidence=0.95
            )
        except Exception as e:
            logger.error(f"出现未知错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    
    
    
    
# ========== 创建应用实例 ==========
api_main = ApiMain()
app = api_main.app

# ========== 直接运行 ==========
if __name__ == "__main__":
    uvicorn.run(
        "src.ai_agent.api.api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
    
    
    
    
    
    
    
    
    