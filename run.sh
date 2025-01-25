#!/bin/bash

# Function to print the usage of the script
function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --generate-traces    Run the generate_traces function."
    echo "  --motion-control     Run the finetuned_motion_control function."
    echo "  --no-clear           Skip clearing the screen at the beginning."
    echo "  --no-clear-pycache   Skip removing __pycache__ directories at the end."
    echo "  --help               Display this help message."
    echo
    echo "Example:"
    echo "  $0 --generate-traces --motion-control"
    exit 1
}

# Default flags for clear and clear-pycache
CLEAR_SCREEN=true
CLEAR_PYCACHE=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --generate-traces)
            GENERATE_TRACES=true
            shift
            ;;
        --motion-control)
            MOTION_CONTROL=true
            shift
            ;;
        --no-clear)
            CLEAR_SCREEN=false
            shift
            ;;
        --no-clear-pycache)
            CLEAR_PYCACHE=false
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Invalid option: $arg"
            usage
            ;;
    esac
done

# Clear the screen if not skipped
if [ "$CLEAR_SCREEN" = true ]; then
    clear
fi

# Run the generate_traces function if requested
if [ "$GENERATE_TRACES" = true ]; then
    echo "Running generate_traces..."
    /home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/trace_2/bin/python -m integration.src.generate_traces
fi

# Run the finetuned_motion_control function if requested
if [ "$MOTION_CONTROL" = true ]; then
    echo "Running finetuned_motion_control..."
    cd priorMDM || exit
    /home/ML_courses/03683533_2024/anton_kfir_daniel/conda/miniconda3/envs/priorMDM_env/bin/python -m sample.finetuned_motion_control \
        --model_path save/root_horizontal_finetuned/model000280000.pt \
        --text_condition "a person is raising hands"
fi

# Remove __pycache__ directories at the end if not skipped
if [ "$CLEAR_PYCACHE" = true ]; then
    echo "Removing all __pycache__ directories..."
    find . -name "__pycache__" -exec rm -r {} +
    echo "All __pycache__ directories have been removed."
fi
