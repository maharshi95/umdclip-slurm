SCRATCH_DIR='/fs/clip-scratch'
MY_CACHE_DIR="${SCRATCH_DIR}/${USER}"

export XDG_CACHE="${MY_CACHE_DIR}/.cache"
export PYSERINI_CACHE="${XDG_CACHE}/pyserini"
export HF_HOME="${XDG_CACHE}/huggingface"
export TORCH_HOME="${XDG_CACHE}/.torch"
export FLAIR_CACHE_ROOT="${XDG_CACHE}/flair"
export PIP_CACHE_DIR="${XDG_CACHE}/.pip"

alias quickgpu="srun --qos=default --time=01:00:00 --mem=30G --gres=gpu:1 --job-name=gpu-1hr --pty zsh"
alias quickcpu="srun --qos=default --time=01:00:00 --mem=30G --job-name=cpu-1hr --pty zsh"

alias sq="squeue -u $USER"

# Function: pf
# Description: Establishes an SSH tunnel to a specified host, forwarding a local port to a remote port.
# Usage: pf <host> [local_port] [remote_port]
# Parameters:
#   host        - The remote host to connect to.
#   local_port  - (Optional) The local port to forward. Defaults to 8888 if not specified.
#   remote_port - (Optional) The remote port to forward to. Defaults to 8888 if not specified.
# Example:
#   pf example.com 8080 9090
#   This will forward local port 8080 to port 9090 on example.com.
function pf() {
  local host=$1
  local local_port="${2:-8888}"
  local remote_port="${3:-8888}"
  ssh -N -f -L localhost:${local_port}:${host}:${remote_port} ${USER}@${host}
}

# This function, pf-cancel, takes a single argument (port number) and checks if any process is using that port.
# If no process is found using the specified port, it prints a message indicating that the port is available for use.
# If a process is found using the specified port, it prints a message indicating the process ID and kills the process.
#
# Usage:
#   pf-cancel <port_number>
#
# Arguments:
#   port_number: The port number to check for any running process.
function pf-cancel() (
  local port=$1
  local ps_id=$(lsof -ti tcp:"$port")
  if [[ -z ${ps_id} ]]; then
    echo "No process running at port ${port}. Port is available for use."
  else
    echo "Killing the process ${ps_id} using port ${port}"
    kill ${ps_id}
  fi
)