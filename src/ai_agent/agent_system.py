import logging
from typing import List

from ai_agent.models.agent_langgraph import AgentLangGraph
from langchain_core.messages import  BaseMessage



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class AgentSystem:

    def __init__(self):

        self.graph = AgentLangGraph()
        self.messages: List[BaseMessage] = []
    """
    多意图识别：目前采用了后置方式，应该采用前置方式【方案在：LangGraph多阶段检索优化】
    新的方向：有新的方案->Kimi K2.5 问题是【我怎么感觉我训练出来的实体识别模型和.......】
    要不要写一个Agent SKill  
    知识图谱要不要加一个实体+关系的查询方法
    """
    
    def execute_question(self,question):
        answer = self.graph.execute(question, "lang_graph_react_agent_2026_7_21")
        return answer

   


if __name__ == '__main__':

    system = AgentSystem()
    
    question_list = [
        "第10、11章发生了什么事情？主要描述谁的？有没有我喜欢的打斗环节",
        "他们之间是什么关系?为什么张元烛救她们",
        "对方是怎么死的？",
        "被救出她们后，张元烛离开后发生了什么事情？",
        "所有的事情是发生在哪个地方？"

    ]
    for question in question_list:
        print(f"\n👤 用户: {question}")
        answer = system.execute_question(question)
        print(f"🤖 助手: {answer}...")
        print("-" * 60)
        








