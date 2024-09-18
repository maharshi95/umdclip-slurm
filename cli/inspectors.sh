# Define an array of search paths for Slurm log files
slogs_search_paths=(
    "$HOME/slurm_logs"
    "./"
)

function slogs() {
    if [ $# -eq 0 ]; then
        echo "Usage: slogs <job_id>"
        return 1
    fi

    local job_id=$1
    local matching_files=()

    # Search for matching files in defined paths
    for search_path in "${slogs_search_paths[@]}"; do
        if [ -d "$search_path" ]; then
            if [ "$search_path" = "./" ]; then
                # Non-recursive search for current directory
                matching_files+=($(find "$search_path" -maxdepth 1 -type f -name "*${job_id}*.out"))
            else
                # Recursive search for other directories
                matching_files+=($(find "$search_path" -type f -name "*${job_id}*.out"))
            fi
        fi
    done

    # Remove non-existent files and duplicates
    if [ -n "$ZSH_VERSION" ]; then
        # Zsh-specific array handling
        matching_files=(${(u)matching_files[@]:#*(0)})
    else
        # Bash-compatible array handling
        matching_files=($(printf "%s\n" "${matching_files[@]}" | sort -u | grep -v '^$'))
    fi

    echo "Matching files: ${matching_files[@]}"

    if [ ${#matching_files[@]} -eq 0 ]; then
        echo "No log files found for job ID $job_id"
        return 1
    elif [ ${#matching_files[@]} -le 5 ]; then
        tail -f "${matching_files[@]}"
    else
        echo "Multiple log files found for job ID $job_id:"
        if [ -n "$ZSH_VERSION" ]; then
            # Zsh-specific handling (1-indexed)
            for i in {1..${#matching_files[@]}}; do
                echo "$i. $(basename "${matching_files[$i]}")"
            done
        else
            # Bash-specific handling (0-indexed)
            for ((i=0; i<${#matching_files[@]}; i++)); do
                echo "$((i+1)). $(basename "${matching_files[$i]}")"
            done
        fi
        if [ -n "$ZSH_VERSION" ]; then
            # Zsh-specific read command
            read "choice?Enter the number of the file you want to tail (or 'a' for all): "
        else
            # Bash-specific read command
            read -p "Enter the number of the file you want to tail (or 'a' for all): " choice
        fi
        if [[ $choice == "a" || $choice == "A" ]]; then
            tail -f "${matching_files[@]}"
        elif [[ $choice =~ '^[0-9]+$' && $choice -ge 1 && $choice -le ${#matching_files[@]} ]]; then
            if [ -n "$ZSH_VERSION" ]; then
                # Zsh-specific array indexing (1-indexed)
                filepath="${matching_files[$choice]}"
            else
                # Bash-specific array indexing (0-indexed)
                filepath="${matching_files[$((choice-1))]}"
            fi
            echo "Tailing $filepath"
            tail -f "$filepath"
        else
            echo "Invalid choice. Please enter a number between 1 and ${#matching_files[@]}, or 'a' for all."
            return 1
        fi
    fi
}