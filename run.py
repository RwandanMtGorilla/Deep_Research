"""Run Deep Research from command line."""

import asyncio
import warnings

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown

from src.research_agent_full import deep_researcher_builder

# ===== FILL IN YOUR RESEARCH QUERY HERE =====
QUERY = "一位欧洲学者的某项开源硬件项目，其灵感源于一个著名的元胞自动机，该项目的一个早期物理设计从四边形框架演变为更稳固的三角形结构。这位在机械工程某一分支领域深耕的学者，从大学教职岗位上引退后，继续领导一个与该项目相关的商业实体。该实体在21世纪10年代中期停止了在其欧洲本土的主要交易，但其在一个亚洲国家的业务得以延续。这个商业实体的英文名称是什么？要求格式形如：Alibaba Group Limited。"
# =============================================

warnings.filterwarnings("ignore")


async def main():
    """Run the deep research agent."""
    checkpointer = InMemorySaver()
    agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1", "recursion_limit": 10}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=QUERY)]},
        config=config,
    )

    console = Console()
    console.print(Markdown(result["final_report"]))


if __name__ == "__main__":
    asyncio.run(main())
