from dotenv import load_dotenv
from neo4j import GraphDatabase
import os



def connction ():
    load_dotenv("")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=('neo4j', 'agent12345'))
    with driver.session() as session:
        session.run('RETURN 1')
    print('Neo4j 连接成功')
    
    
if __name__ == "__main__":
    connction()