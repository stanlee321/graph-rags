# Building-an-Advanced-RAG-Chatbot-with-Knowledge-Graphs


## Neo4J init.

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS=\[\"apoc\"\]  \
    neo4j:latest
```

## Libs

```bash
python -m spacy download en_core_web_lg
```

```bash
pip install -U langchain-neo4j
```


## Ref  
- https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb