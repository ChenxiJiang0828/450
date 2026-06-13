import sys
import os
import time
import logging
import uuid
import asyncio
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import tsmrt.manager as dialog_manager
from tsmrt.tsm_input import TsmInput, Session, DialogHistory, DialogOutput, SpeakOutput, Functions, FunctionRequest, RequestBody
from tsmrt.tsm_output import TsmOutput
import json
import pdb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDialogManager:
    def __init__(self, instances, c_info=None):
        self.dialog_manager = dialog_manager.Dialog_Manager(instances, c_info)
        self.session = Session(dialogHistory=[])

    def update_dialog_state(self, tsm_input: TsmInput):
        self.session = Session(dialogHistory=[])
        if tsm_input.session and tsm_input.session.dialogHistory:
            self.session.dialogHistory = tsm_input.session.dialogHistory.copy()

    def add_dialog_history(self, user_input: str, nlg_output: str = "", function_name: str = "", skill: str = "", source: str = "", data_source: str = ""):
        dialog_output = None
        if nlg_output:
            dialog_output = DialogOutput(speak=SpeakOutput(text=nlg_output))
        
        functions = None
        if function_name:
            functions = Functions(request=FunctionRequest(name=function_name))
        
        dialog_history = DialogHistory(
            timestamp=int(time.time() * 1000),
            input=user_input,
            skill=skill,
            source=source,
            output=dialog_output,
            functions=functions,
            dataSource=data_source
        )
        
        if self.session.dialogHistory is None:
            self.session.dialogHistory = []
        self.session.dialogHistory.append(dialog_history)

    def get_dialog_history(self) -> List[DialogHistory]:
        return self.session.dialogHistory if self.session.dialogHistory else []

    async def chat_process(self, user_input: str) -> TsmOutput:
        request_id = str(uuid.uuid4())
        request_body = RequestBody(input=user_input, requestId=request_id, recordId=request_id)
        tsm_input = TsmInput(request=request_body, session=self.session)
        
        tsm_output = await self.dialog_manager.chat_process(tsm_input)
        
        dlg_input = tsm_output.request.text
        dlg_nlg = "好的"
        if len(tsm_output.result.tsmResult) > 0:
            dlg_func = tsm_output.result.tsmResult[0].functions.request.name or ""
        else:
            dlg_func = ""
        self.add_dialog_history(dlg_input, dlg_nlg, dlg_func)

        return tsm_output


c_instance = {
    "_description": "智能座舱",
    "all_functions": [json.loads(it.strip()) for it in open("../../function_call_data/car_control_func.jsonl", "r").readlines()] +
#                     [json.loads(it.strip()) for it in open("../../function_call_data/map_func.jsonl", "r").readlines()] +
                    [json.loads(it.strip()) for it in open("../../function_call_data/multimedia_func.jsonl", "r").readlines()]
}
c_instances = [c_instance]
# c_instances = None

test_dm = TestDialogManager(c_instances)
test_dm.dialog_manager.l2_classifier.model = "DFM2-Pro-Spring"
test_dm.dialog_manager.function_call_agent.model = "dfm3_tsl_built_in_1.0"
# test_dm.dialog_manager.function_call_agent.model = "DFM3.5-Turbo-Sep"
# test_dm.dialog_manager.function_call_agent.model = "DFM3.5-Turbo"
# test_dm.dialog_manager.function_call_agent.model = "deepseek-v4-flash"
test_dm.dialog_manager.used_rag_search = True
test_dm.dialog_manager.rag_topk = 10

async def main():
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        tsm_output = await test_dm.chat_process(user_input)
        tsm_output = tsm_output.model_dump(exclude_none=True)
        logger.info(f"process-output: {json.dumps(tsm_output, ensure_ascii=False)}")

asyncio.run(main())
