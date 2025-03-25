docker run \
     --name testneo4j \
     -p7474:7474 -p7687:7687 \
     -d \
     -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* \
     -v $HOME/neo4j/data:/data \
     -v $HOME/neo4j/logs:/logs \
     -v $HOME/neo4j/import:/var/lib/neo4j/import \
     -v $HOME/neo4j/plugins:/plugins \
     --env NEO4J_AUTH=neo4j/test \
     --env NEO4J_PLUGINS='["graph-data-science"]' \
     neo4j:latest