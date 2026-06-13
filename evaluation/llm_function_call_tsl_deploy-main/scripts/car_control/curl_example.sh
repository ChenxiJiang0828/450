#!/usr/bin/env bash
#
# Usage: 
# Author: sj123(sheng.jiang@aispeech.com)


test_case="切换大模型"
curl -X POST "http://tsmserver.hd-public.beta.duiopen.com/tsm/v1/functions?productId=279628310&productVersion=1&aliasKey=prod" \
  -H "Content-Type: application/json" \
  -d '{"request":{"input":"'"$test_case"'"},"context":{"product":{"productId":"123","productVersion":"1"},"skills":[{"id":"2018040200000004","name":"车载控制","task":"车身控制","version":"3","useTsm":true,"tsm":{"domain":"CarControl"}}]}}'
