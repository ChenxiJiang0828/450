#!/usr/bin/env bash
# 网络/代理：登录节点可走本地代理，GPU 计算节点必须直连（127.0.0.1:13000 在计算节点不可用）

disable_proxy() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
}

enable_login_proxy_if_available() {
  if curl -sf --connect-timeout 2 http://127.0.0.1:13000 >/dev/null 2>&1; then
    export http_proxy=http://127.0.0.1:13000
    export https_proxy=http://127.0.0.1:13000
    echo "使用登录节点代理: ${http_proxy}"
  else
    disable_proxy
  fi
}

is_compute_node() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || [[ -n "${SLURM_NODELIST:-}" ]]
}

configure_network_for_current_node() {
  if is_compute_node; then
    disable_proxy
    echo "计算节点：已关闭代理，使用直连/集群网络"
  else
    enable_login_proxy_if_available
  fi
}
