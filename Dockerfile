# Use the official Neo4j image
FROM neo4j

# Expose Neo4j ports
EXPOSE 7474 7687

ENV NEO4J_AUTH=neo4j/newPassword

# Default command to run Neo4j
CMD ["neo4j"]

