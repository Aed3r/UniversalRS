#!/bin/bash
echo "Executing command: python3 GPRS/test.py"
START_TIME=$(date +%s)
python3 GPRS/test.py
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Command executed in $DURATION seconds"
