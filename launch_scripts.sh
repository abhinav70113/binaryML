#!/bin/bash

# List of commands to run
commands=(
  "sbatch tuner_attention_z.sh"
  "sbatch tuner_attention_f.sh"
  "sbatch tuner_attention_f.sh"
  "sbatch tuner_attention_f.sh"
)

# Function to execute a command and retry if it fails
execute_command_with_retry() {
  local command=$1
  local max_retries=5000000
  local retry_interval=600  # 10 minutes in seconds
  local retries=0
  local exit_code=1

  while [ $exit_code -ne 0 ] && [ $retries -lt $max_retries ]; do
    echo "Executing command: $command"
    output=$($command 2>&1)
    exit_code=$?

    if [[ $output == *"sbatch: error: AssocMaxSubmitJobLimit"* || $output == *"sbatch: error: Batch job submission failed: Job violates accounting/QOS policy"* ]]; then
      echo "Job submission limit or resource allocation policy violated. Retrying in 10 minutes..."
      retries=$((retries + 1))
      sleep $retry_interval
    elif [ $exit_code -ne 0 ]; then
      echo "Command failed. Retrying in 10 minutes..."
      retries=$((retries + 1))
      sleep $retry_interval
    fi
  done

  if [ $retries -eq $max_retries ]; then
    echo "Maximum retries reached for command: $command"
  fi
}

# Loop through the commands and execute them with retry
for cmd in "${commands[@]}"; do
  execute_command_with_retry "$cmd"
done
