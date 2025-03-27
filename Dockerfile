FROM neo4j:5.12.0

# Expose Neo4j ports
EXPOSE 7474 7687

# Set authentication
ENV NEO4J_AUTH=neo4j/newPassword

# Install plugins
ENV NEO4J_PLUGINS=["graph-data-science"]

# Create a volume to persist data
VOLUME ["/data", "/plugins"]

# Default command to run Neo4j
CMD ["neo4j"]