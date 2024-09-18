# Setting up `workon` function to cd into project directory and activate conda env

# Add project name and workspace directory to the dictionaries
declare -A WORKON_PROJ_DIRS=(
    [proj1]="path/to/proj1-workspace-dir"
    [proj2]="path/to/proj2-workspace-dir"
)

# Add project name and conda environment name to the dictionaries
declare -A WORKON_CONDA_ENVS=(
    [proj1]="proj1-conda-env"
    [proj2]="proj2-conda-env"
)

function workon() {
    local proj_name=$1
    local project_dir=${WORKON_PROJ_DIRS[$proj_name]}
    local project_env=${WORKON_CONDA_ENVS[$proj_name]}

    if [[ -z ${project_dir} ]]; 
    then 
        echo "Project '$proj_name' not found"
        echo "Available projects: ${!WORKON_PROJ_DIRS[@]}"
        return 1
    fi

    cd "${project_dir}" || return 1

    if [[ ${project_env} ]]; 
    then 
        echo "Activating conda environment '$project_env'"
        conda activate ${project_env}
    else
        echo "No named conda environment found for project '$proj_name'. Looking for ${project_dir}/env/";
        if [[ -d "${project_dir}/env" ]]; then
            conda activate "${project_dir}/env"
        else
            echo "No 'env' directory found in project directory. Unable to activate conda environment."
            return 1
        fi
    fi

    if [[ $? -ne 0 ]]; then
        echo "Failed to activate conda environment."
        return 1
    fi

    if [[ -f "${project_dir}/.env" ]]; then
        echo "Found .env file in ${project_dir}. Loading environment variables."
        set -a
        source "${project_dir}/.env"
        set +a
    else
        echo "No .env file found in project directory. Just adding project directory to PYTHONPATH"
        export PYTHONPATH="${project_dir}:${PYTHONPATH}"
    fi

    echo "Project '$proj_name' activated."
    echo "PYTHONPATH items:"
    echo "$PYTHONPATH" | tr ":" "\n" | nl
    export PYTHONPATH
}
