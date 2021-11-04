#!/bin/bash
#
# プロジェクトで利用するコマンドを記載するスクリプト。

set -eu


readonly SCRIPT_DIR=$(cd $(dirname $0); pwd)
readonly SCRIPT_NAME=$(basename $0)
#readonly PROJECT_DIR=$(cd $SCRIPT_DIR/../..; pwd)
readonly PROJECT_DIR=$(pwd)
readonly DATA_DIR=${DATA_DIR:-"$(cd $PROJECT_DIR; pwd)"}

readonly JUPYTER_CONFIG="$PROJECT_DIR/src/scripts/jupyter_notebook_config.py"

# HELP.
function _usage() {
  cat <<EOF
$SCRIPT_NAME is a tool for failure prediction.

Usage:
$SCRIPT_NAME [command] [options]

Commands:
jupyter: run jupyter server 
EOF
}

# jupyter 
function _jupyter() {
    readonly local ENVFILE="$PROJECT_DIR/.env"
    env $(cat ${ENVFILE} | xargs) poetry run jupyter lab --config $JUPYTER_CONFIG
}

# python
function _python() {
    readonly local ENVFILE="$PROJECT_DIR/.env"
    readonly PYTHON_SCRIPT=$*
    env $(cat ${ENVFILE} | xargs) poetry run python $PYTHON_SCRIPT
}


readonly SUB_COMMAND=${1}
shift
readonly OPTIONS=$*
case "$SUB_COMMAND" in
  "help" ) _usage;;
  "jupyter" ) _jupyter $OPTIONS;;
  "python" ) _python $OPTIONS;;
esac