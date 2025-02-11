#!/bin/bash
# Usage: bash identify.sh <train_graphs_path> <train_labels_path> <subgraphs_output_path>
#!/bin/bash

CMD="python3 identify.py \"$1\" \"$2\" \"$3\""

# Check if $4 is provided
if [ -n "$4" ]; then
    CMD="$CMD --m \"$4\""
fi

# Check if $5 is provided
if [ -n "$5" ]; then
    CMD="$CMD --s \"$5\""
fi

# Execute the command
eval $CMD

