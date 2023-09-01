#!/bin/bash
echo "Executing command: python3 ./GPRS/KnowledgeGraphEmbeddings/kge-ml100k.py"
START_TIME=$(date +%s)
python3 ./GPRS/KnowledgeGraphEmbeddings/kge-ml100k.py
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Command executed in $DURATION seconds"
