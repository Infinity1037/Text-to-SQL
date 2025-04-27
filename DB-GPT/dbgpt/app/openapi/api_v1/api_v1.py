import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import Executor
from io import BytesIO
from typing import List, Optional, cast, Dict

import aiofiles
import chardet
import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from dbgpt._private.config import Config
from dbgpt.app.knowledge.request.request import KnowledgeSpaceRequest
from dbgpt.app.knowledge.service import KnowledgeService
from dbgpt.app.openapi.api_view_model import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatSceneVo,
    ConversationVo,
    DeltaMessage,
    MessageVo,
    Result,
)
from dbgpt.app.scene import BaseChat, ChatFactory, ChatScene
from dbgpt.component import ComponentType
from dbgpt.configs import TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE
from dbgpt.configs.model_config import KNOWLEDGE_UPLOAD_ROOT_PATH
from dbgpt.core.awel import BaseOperator, CommonLLMHttpRequestBody
from dbgpt.core.awel.dag.dag_manager import DAGManager
from dbgpt.core.awel.util.chat_util import safe_chat_stream_with_dag_task
from dbgpt.core.interface.message import OnceConversation
from dbgpt.datasource.db_conn_info import DBConfig, DbTypeInfo
from dbgpt.model.base import FlatSupportedModel
from dbgpt.model.cluster import BaseModelController, WorkerManager, WorkerManagerFactory
from dbgpt.rag.summary.db_summary_client import DBSummaryClient
from dbgpt.serve.agent.db.gpts_app import UserRecentAppsDao, adapt_native_app_model
from dbgpt.serve.flow.service.service import Service as FlowService
from dbgpt.serve.utils.auth import UserRequest, get_user_from_headers
from dbgpt.util.executor_utils import (
    DefaultExecutorFactory,
    ExecutorFactory,
    blocking_func_to_async,
)
from dbgpt.util.file_client import FileClient
from dbgpt.util.tracer import SpanType, root_tracer
from pydantic import BaseModel, Field
import re
import html

router = APIRouter()
CFG = Config()
CHAT_FACTORY = ChatFactory()
logger = logging.getLogger(__name__)
knowledge_service = KnowledgeService()

model_semaphore = None
global_counter = 0


user_recent_app_dao = UserRecentAppsDao()


def __get_conv_user_message(conversations: dict):
    messages = conversations["messages"]
    for item in messages:
        if item["type"] == "human":
            return item["data"]["content"]
    return ""


def __new_conversation(chat_mode, user_name: str, sys_code: str) -> ConversationVo:
    unique_id = uuid.uuid1()
    return ConversationVo(
        conv_uid=str(unique_id),
        chat_mode=chat_mode,
        user_name=user_name,
        sys_code=sys_code,
    )


def get_db_list(user_id: str = None):
    dbs = CFG.local_db_manager.get_db_list(user_id=user_id)
    db_params = []
    for item in dbs:
        params: dict = {}
        params.update({"param": item["db_name"]})
        params.update({"type": item["db_type"]})
        db_params.append(params)
    return db_params


def plugins_select_info():
    plugins_infos: dict = {}
    for plugin in CFG.plugins:
        plugins_infos.update({f"【{plugin._name}】=>{plugin._description}": plugin._name})
    return plugins_infos


def get_db_list_info(user_id: str = None):
    dbs = CFG.local_db_manager.get_db_list(user_id=user_id)
    params: dict = {}
    for item in dbs:
        comment = item["comment"]
        if comment is not None and len(comment) > 0:
            params.update({item["db_name"]: comment})
    return params


def knowledge_list_info():
    """return knowledge space list"""
    params: dict = {}
    request = KnowledgeSpaceRequest()
    spaces = knowledge_service.get_knowledge_space(request)
    for space in spaces:
        params.update({space.name: space.desc})
    return params


def knowledge_list(user_id: str = None):
    """return knowledge space list"""
    request = KnowledgeSpaceRequest(user_id=user_id)
    spaces = knowledge_service.get_knowledge_space(request)
    space_list = []
    for space in spaces:
        params: dict = {}
        params.update({"param": space.name})
        params.update({"type": "space"})
        params.update({"space_id": space.id})
        space_list.append(params)
    return space_list


def get_chat_flow() -> FlowService:
    """Get Chat Flow Service."""
    return FlowService.get_instance(CFG.SYSTEM_APP)


def get_model_controller() -> BaseModelController:
    controller = CFG.SYSTEM_APP.get_component(
        ComponentType.MODEL_CONTROLLER, BaseModelController
    )
    return controller


def get_worker_manager() -> WorkerManager:
    worker_manager = CFG.SYSTEM_APP.get_component(
        ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
    ).create()
    return worker_manager


def get_dag_manager() -> DAGManager:
    """Get the global default DAGManager"""
    return DAGManager.get_instance(CFG.SYSTEM_APP)


def get_chat_flow() -> FlowService:
    """Get Chat Flow Service."""
    return FlowService.get_instance(CFG.SYSTEM_APP)


def get_executor() -> Executor:
    """Get the global default executor"""
    return CFG.SYSTEM_APP.get_component(
        ComponentType.EXECUTOR_DEFAULT,
        ExecutorFactory,
        or_register_component=DefaultExecutorFactory,
    ).create()


@router.get("/v1/chat/db/list", response_model=Result)
async def db_connect_list(
    db_name: Optional[str] = Query(default=None, description="database name"),
    user_info: UserRequest = Depends(get_user_from_headers),
):
    results = CFG.local_db_manager.get_db_list(
        db_name=db_name, user_id=user_info.user_id
    )
    # 排除部分数据库不允许用户访问
    if results and len(results):
        results = [
            d
            for d in results
            if d.get("db_name") not in ["auth", "dbgpt", "test", "public"]
        ]
    return Result.succ(results)


@router.post("/v1/chat/db/add", response_model=Result)
async def db_connect_add(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ(CFG.local_db_manager.add_db(db_config, user_token.user_id))


@router.get("/v1/permission/db/list", response_model=Result[List])
async def permission_db_list(
    db_name: str = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ()


@router.post("/v1/chat/db/edit", response_model=Result)
async def db_connect_edit(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    return Result.succ(CFG.local_db_manager.edit_db(db_config))


@router.post("/v1/chat/db/delete", response_model=Result[bool])
async def db_connect_delete(db_name: str = None):
    if not db_name:
         logger.error("db_connect_delete called with no db_name.")
         return Result.failed(code="E2001", msg="Database name cannot be empty.")
    try:
        # 尝试删除 profile，如果失败则记录警告
        logger.info(f"Attempting to delete profile for database: {db_name}")
        # Assuming delete_db_profile might not exist or could fail gracefully
        if hasattr(CFG.local_db_manager.db_summary_client, 'delete_db_profile'):
             CFG.local_db_manager.db_summary_client.delete_db_profile(db_name)
             logger.info(f"Successfully called delete_db_profile for database: {db_name} (Actual deletion depends on implementation)")
        else:
            logger.warning(f"db_summary_client does not have delete_db_profile method. Skipping profile deletion for {db_name}.")

    except Exception as e:
        # 记录警告，但继续执行删除连接配置的操作
        logger.warning(f"Could not delete profile for database {db_name}. Error: {e}. Proceeding to delete connection config.", exc_info=True) # Add exc_info for more details

    # 删除连接配置本身
    try:
        logger.info(f"Attempting to delete connection config for database: {db_name}")
        # Assuming delete_db returns True on success, False or raises exception on failure
        delete_success = CFG.local_db_manager.delete_db(db_name)
        if delete_success:
             logger.info(f"Successfully deleted connection config for database: {db_name}")
             return Result.succ(True)
        else:
             # If delete_db returns False without exception, treat as failure
             logger.error(f"Failed to delete connection config for database: {db_name} (delete_db returned False).")
             return Result.failed(code="E2002", msg=f"Failed to delete connection config for {db_name}")
    except Exception as e:
         logger.error(f"Error deleting connection config for database {db_name}: {e}", exc_info=True)
         # Return failure with specific error code and message
         return Result.failed(code="E2003", msg=f"Exception occurred while deleting connection config for {db_name}: {str(e)}")


@router.post("/v1/chat/db/refresh", response_model=Result[bool])
async def db_connect_refresh(db_config: DBConfig = Body()):
    CFG.local_db_manager.db_summary_client.delete_db_profile(db_config.db_name)
    success = await CFG.local_db_manager.async_db_summary_embedding(
        db_config.db_name, db_config.db_type
    )
    return Result.succ(success)


async def async_db_summary_embedding(db_name, db_type):
    db_summary_client = DBSummaryClient(system_app=CFG.SYSTEM_APP)
    db_summary_client.db_summary_embedding(db_name, db_type)


@router.post("/v1/chat/db/test/connect", response_model=Result[bool])
async def test_connect(
    db_config: DBConfig = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    try:
        # TODO Change the synchronous call to the asynchronous call
        CFG.local_db_manager.test_connect(db_config)
        return Result.succ(True)
    except Exception as e:
        return Result.failed(code="E1001", msg=str(e))


@router.post("/v1/chat/db/summary", response_model=Result[bool])
async def db_summary(db_name: str, db_type: str):
    # TODO Change the synchronous call to the asynchronous call
    async_db_summary_embedding(db_name, db_type)
    return Result.succ(True)


@router.get("/v1/chat/db/support/type", response_model=Result[List[DbTypeInfo]])
async def db_support_types():
    support_types = CFG.local_db_manager.get_all_completed_types()
    db_type_infos = []
    for type in support_types:
        db_type_infos.append(
            DbTypeInfo(db_type=type.value(), is_file_db=type.is_file_db())
        )
    return Result[DbTypeInfo].succ(db_type_infos)


@router.post("/v1/chat/dialogue/scenes", response_model=Result[List[ChatSceneVo]])
async def dialogue_scenes(user_info: UserRequest = Depends(get_user_from_headers)):
    scene_vos: List[ChatSceneVo] = []
    new_modes: List[ChatScene] = [
        ChatScene.ChatWithDbExecute,
        ChatScene.ChatWithDbQA,
        ChatScene.ChatExcel,
        ChatScene.ChatKnowledge,
        ChatScene.ChatDashboard,
        ChatScene.ChatAgent,
    ]
    for scene in new_modes:
        scene_vo = ChatSceneVo(
            chat_scene=scene.value(),
            scene_name=scene.scene_name(),
            scene_describe=scene.describe(),
            param_title=",".join(scene.param_types()),
            show_disable=scene.show_disable(),
        )
        scene_vos.append(scene_vo)
    return Result.succ(scene_vos)


@router.post("/v1/resource/params/list", response_model=Result[List[dict]])
async def resource_params_list(
    resource_type: str,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    if resource_type == "database":
        result = get_db_list()
    elif resource_type == "knowledge":
        result = knowledge_list()
    elif resource_type == "tool":
        result = plugins_select_info()
    else:
        return Result.succ()
    return Result.succ(result)


@router.post("/v1/chat/mode/params/list", response_model=Result[List[dict]])
async def params_list(
    chat_mode: str = ChatScene.ChatNormal.value(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    if ChatScene.ChatWithDbQA.value() == chat_mode:
        result = get_db_list()
    elif ChatScene.ChatWithDbExecute.value() == chat_mode:
        result = get_db_list()
    elif ChatScene.ChatDashboard.value() == chat_mode:
        result = get_db_list()
    elif ChatScene.ChatExecution.value() == chat_mode:
        result = plugins_select_info()
    elif ChatScene.ChatKnowledge.value() == chat_mode:
        result = knowledge_list()
    elif ChatScene.ChatKnowledge.ExtractRefineSummary.value() == chat_mode:
        result = knowledge_list()
    else:
        return Result.succ()
    return Result.succ(result)


@router.post("/v1/resource/file/upload")
async def file_upload(
    chat_mode: str,
    conv_uid: str,
    sys_code: Optional[str] = None,
    model_name: Optional[str] = None,
    doc_file: UploadFile = File(...),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"file_upload:{conv_uid},{doc_file.filename}")
    file_client = FileClient()
    file_name = doc_file.filename
    is_oss, file_key = await file_client.write_file(
        conv_uid=conv_uid, doc_file=doc_file
    )

    _, file_extension = os.path.splitext(file_name)
    if file_extension.lower() in [".xls", ".xlsx", ".csv"]:
        file_param = {
            "is_oss": is_oss,
            "file_path": file_key,
            "file_name": file_name,
            "file_learning": True,
        }
        # Prepare the chat
        dialogue = ConversationVo(
            conv_uid=conv_uid,
            chat_mode=chat_mode,
            select_param=file_param,
            model_name=model_name,
            user_name=user_token.user_id,
            sys_code=sys_code,
        )
        chat: BaseChat = await get_chat_instance(dialogue)
        await chat.prepare()

        # Refresh messages
        return Result.succ(file_param)
    else:
        return Result.succ(
            {
                "is_oss": is_oss,
                "file_path": file_key,
                "file_learning": False,
                "file_name": file_name,
            }
        )


@router.post("/v1/resource/file/delete")
async def file_delete(
    conv_uid: str,
    file_key: str,
    user_name: Optional[str] = None,
    sys_code: Optional[str] = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"file_delete:{conv_uid},{file_key}")
    oss_file_client = FileClient()

    return Result.succ(
        await oss_file_client.delete_file(conv_uid=conv_uid, file_key=file_key)
    )


@router.post("/v1/resource/file/read")
async def file_read(
    conv_uid: str,
    file_key: str,
    user_name: Optional[str] = None,
    sys_code: Optional[str] = None,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"file_read:{conv_uid},{file_key}")
    file_client = FileClient()
    res = await file_client.read_file(conv_uid=conv_uid, file_key=file_key)
    df = pd.read_excel(res, index_col=False)
    return Result.succ(df.to_json(orient="records", date_format="iso", date_unit="s"))


def get_hist_messages(conv_uid: str, user_name: str = None):
    from dbgpt.serve.conversation.serve import Service as ConversationService

    instance: ConversationService = ConversationService.get_instance(CFG.SYSTEM_APP)
    return instance.get_history_messages({"conv_uid": conv_uid, "user_name": user_name})


async def get_chat_instance(dialogue: ConversationVo = Body()) -> BaseChat:
    logger.info(f"get_chat_instance:{dialogue}")
    if not dialogue.chat_mode:
        dialogue.chat_mode = ChatScene.ChatNormal.value()
    if not dialogue.conv_uid:
        conv_vo = __new_conversation(
            dialogue.chat_mode, dialogue.user_name, dialogue.sys_code
        )
        dialogue.conv_uid = conv_vo.conv_uid

    if not ChatScene.is_valid_mode(dialogue.chat_mode):
        raise StopAsyncIteration(
            Result.failed("Unsupported Chat Mode," + dialogue.chat_mode + "!")
        )

    chat_param = {
        "chat_session_id": dialogue.conv_uid,
        "user_name": dialogue.user_name,
        "sys_code": dialogue.sys_code,
        "current_user_input": dialogue.user_input,
        "select_param": dialogue.select_param,
        "model_name": dialogue.model_name,
        "app_code": dialogue.app_code,
        "ext_info": dialogue.ext_info,
        "temperature": dialogue.temperature,
    }
    chat: BaseChat = await blocking_func_to_async(
        get_executor(),
        CHAT_FACTORY.get_implementation,
        dialogue.chat_mode,
        **{"chat_param": chat_param},
    )
    return chat


@router.post("/v1/chat/prepare")
async def chat_prepare(
    dialogue: ConversationVo = Body(),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(json.dumps(dialogue.__dict__))
    # dialogue.model_name = CFG.LLM_MODEL
    dialogue.user_name = user_token.user_id if user_token else dialogue.user_name
    logger.info(f"chat_prepare:{dialogue}")
    ## check conv_uid
    chat: BaseChat = await get_chat_instance(dialogue)

    await chat.prepare()

    # Refresh messages
    return Result.succ(get_hist_messages(dialogue.conv_uid, user_token.user_id))


@router.post("/v1/chat/completions")
async def chat_completions(
    dialogue: ConversationVo = Body(),
    flow_service: FlowService = Depends(get_chat_flow),
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(
        f"chat_completions:{dialogue.chat_mode},{dialogue.select_param},{dialogue.model_name}, timestamp={int(time.time() * 1000)}"
    )
    dialogue.user_name = user_token.user_id if user_token else dialogue.user_name
    dialogue = adapt_native_app_model(dialogue)
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
    }
    try:
        domain_type = _parse_domain_type(dialogue)
        if dialogue.chat_mode == ChatScene.ChatAgent.value():
            from dbgpt.serve.agent.agents.controller import multi_agents

            dialogue.ext_info.update({"model_name": dialogue.model_name})
            dialogue.ext_info.update({"incremental": dialogue.incremental})
            dialogue.ext_info.update({"temperature": dialogue.temperature})
            return StreamingResponse(
                multi_agents.app_agent_chat(
                    conv_uid=dialogue.conv_uid,
                    gpts_name=dialogue.app_code,
                    user_query=dialogue.user_input,
                    user_code=dialogue.user_name,
                    sys_code=dialogue.sys_code,
                    **dialogue.ext_info,
                ),
                headers=headers,
                media_type="text/event-stream",
            )
        elif dialogue.chat_mode == ChatScene.ChatFlow.value():
            flow_req = CommonLLMHttpRequestBody(
                model=dialogue.model_name,
                messages=dialogue.user_input,
                stream=True,
                # context=flow_ctx,
                # temperature=
                # max_new_tokens=
                # enable_vis=
                conv_uid=dialogue.conv_uid,
                span_id=root_tracer.get_current_span_id(),
                chat_mode=dialogue.chat_mode,
                chat_param=dialogue.select_param,
                user_name=dialogue.user_name,
                sys_code=dialogue.sys_code,
                incremental=dialogue.incremental,
            )
            return StreamingResponse(
                flow_service.chat_stream_flow_str(dialogue.select_param, flow_req),
                headers=headers,
                media_type="text/event-stream",
            )
        elif domain_type is not None and domain_type != "Normal":
            return StreamingResponse(
                chat_with_domain_flow(dialogue, domain_type),
                headers=headers,
                media_type="text/event-stream",
            )

        else:
            with root_tracer.start_span(
                "get_chat_instance", span_type=SpanType.CHAT, metadata=dialogue.dict()
            ):
                chat: BaseChat = await get_chat_instance(dialogue)

            if not chat.prompt_template.stream_out:
                return StreamingResponse(
                    no_stream_generator(chat),
                    headers=headers,
                    media_type="text/event-stream",
                )
            else:
                return StreamingResponse(
                    stream_generator(chat, dialogue.incremental, dialogue.model_name),
                    headers=headers,
                    media_type="text/plain",
                )
    except Exception as e:
        logger.exception(f"Chat Exception!{dialogue}", e)

        async def error_text(err_msg):
            yield f"data:{err_msg}\n\n"

        return StreamingResponse(
            error_text(str(e)),
            headers=headers,
            media_type="text/plain",
        )
    finally:
        # write to recent usage app.
        if dialogue.user_name is not None and dialogue.app_code is not None:
            user_recent_app_dao.upsert(
                user_code=dialogue.user_name,
                sys_code=dialogue.sys_code,
                app_code=dialogue.app_code,
            )


@router.post("/v1/chat/topic/terminate")
async def terminate_topic(
    conv_id: str,
    round_index: int,
    user_token: UserRequest = Depends(get_user_from_headers),
):
    logger.info(f"terminate_topic:{conv_id},{round_index}")
    try:
        from dbgpt.serve.agent.agents.controller import multi_agents

        return Result.succ(await multi_agents.topic_terminate(conv_id))
    except Exception as e:
        logger.exception("Topic terminate error!")
        return Result.failed(code="E0102", msg=str(e))


@router.get("/v1/model/types")
async def model_types(controller: BaseModelController = Depends(get_model_controller)):
    logger.info(f"/controller/model/types")
    try:
        types = set()
        models = await controller.get_all_instances(healthy_only=True)
        for model in models:
            worker_name, worker_type = model.model_name.split("@")
            if worker_type == "llm" and worker_name not in [
                "codegpt_proxyllm",
                "text2sql_proxyllm",
            ]:
                types.add(worker_name)
        return Result.succ(list(types))

    except Exception as e:
        return Result.failed(code="E000X", msg=f"controller model types error {e}")


@router.get("/v1/test")
async def test():
    return "service status is UP"


@router.get("/v1/model/supports")
async def model_supports(worker_manager: WorkerManager = Depends(get_worker_manager)):
    logger.info(f"/controller/model/supports")
    try:
        models = await worker_manager.supported_models()
        return Result.succ(FlatSupportedModel.from_supports(models))
    except Exception as e:
        return Result.failed(code="E000X", msg=f"Fetch supportd models error {e}")


async def flow_stream_generator(func, incremental: bool, model_name: str):
    stream_id = f"chatcmpl-{str(uuid.uuid1())}"
    previous_response = ""
    async for chunk in func:
        if chunk:
            msg = chunk.replace("\ufffd", "")
            if incremental:
                incremental_output = msg[len(previous_response) :]
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=incremental_output),
                )
                chunk = ChatCompletionStreamResponse(
                    id=stream_id, choices=[choice_data], model=model_name
                )
                yield f"data: {json.dumps(chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n"
            else:
                # TODO generate an openai-compatible streaming responses
                msg = msg.replace("\n", "\\n")
                yield f"data:{msg}\n\n"
            previous_response = msg
    if incremental:
        yield "data: [DONE]\n\n"


async def no_stream_generator(chat):
    with root_tracer.start_span("no_stream_generator"):
        msg = await chat.nostream_call()
        yield f"data: {msg}\n\n"


async def stream_generator(chat, incremental: bool, model_name: str):
    """Generate streaming responses

    Our goal is to generate an openai-compatible streaming responses.
    Currently, the incremental response is compatible, and the full response will be transformed in the future.

    Args:
        chat (BaseChat): Chat instance.
        incremental (bool): Used to control whether the content is returned incrementally or in full each time.
        model_name (str): The model name

    Yields:
        _type_: streaming responses
    """
    span = root_tracer.start_span("stream_generator")
    msg = "[LLM_ERROR]: llm server has no output, maybe your prompt template is wrong."

    stream_id = f"chatcmpl-{str(uuid.uuid1())}"
    previous_response = ""

    # 流式输出
    # async for chunk in chat.stream_call():

    async for chunk in chat.stream_call_db():
        if chunk:
            msg = chunk.replace("\ufffd", "")
            if incremental:
                incremental_output = msg[len(previous_response) :]
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=incremental_output),
                )
                chunk = ChatCompletionStreamResponse(
                    id=stream_id, choices=[choice_data], model=model_name
                )
                yield f"data:{json.dumps(chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n"
            else:
                # TODO generate an openai-compatible streaming responses
                msg = msg.replace("\n", "\\n")
                msg = msg.replace("```", "")

                # 输出格式化
                msg = msg.replace('json ','')
                # msg = msg.replace('"thought"','')
                yield f"data:{msg}\n\n"
            previous_response = msg
            await asyncio.sleep(0.02)
    if incremental:
        yield "data: [DONE]\n\n"
    span.end()


def message2Vo(message: dict, order, model_name) -> MessageVo:
    return MessageVo(
        role=message["type"],
        context=message["data"]["content"],
        order=order,
        model_name=model_name,
    )


def _parse_domain_type(dialogue: ConversationVo) -> Optional[str]:
    if dialogue.chat_mode == ChatScene.ChatKnowledge.value():
        # Supported in the knowledge chat
        if dialogue.app_code == "" or dialogue.app_code == "chat_knowledge":
            spaces = knowledge_service.get_knowledge_space(
                KnowledgeSpaceRequest(name=dialogue.select_param)
            )
        else:
            spaces = knowledge_service.get_knowledge_space(
                KnowledgeSpaceRequest(id=dialogue.select_param)
            )
        if len(spaces) == 0:
            raise ValueError(f"Knowledge space {dialogue.select_param} not found")
        dialogue.select_param = spaces[0].name
        if spaces[0].domain_type:
            return spaces[0].domain_type
    else:
        return None


async def chat_with_domain_flow(dialogue: ConversationVo, domain_type: str):
    """Chat with domain flow"""
    dag_manager = get_dag_manager()
    dags = dag_manager.get_dags_by_tag(TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE, domain_type)
    if not dags or not dags[0].leaf_nodes:
        raise ValueError(f"Cant find the DAG for domain type {domain_type}")

    end_task = cast(BaseOperator, dags[0].leaf_nodes[0])
    space = dialogue.select_param
    connector_manager = CFG.local_db_manager
    # TODO: Some flow maybe not connector
    db_list = [item["db_name"] for item in connector_manager.get_db_list()]
    db_names = [item for item in db_list if space in item]
    if len(db_names) == 0:
        raise ValueError(f"fin repost dbname {space}_fin_report not found.")
    flow_ctx = {"space": space, "db_name": db_names[0]}
    request = CommonLLMHttpRequestBody(
        model=dialogue.model_name,
        messages=dialogue.user_input,
        stream=True,
        extra=flow_ctx,
        conv_uid=dialogue.conv_uid,
        span_id=root_tracer.get_current_span_id(),
        chat_mode=dialogue.chat_mode,
        chat_param=dialogue.select_param,
        user_name=dialogue.user_name,
        sys_code=dialogue.sys_code,
        incremental=dialogue.incremental,
    )
    async for output in safe_chat_stream_with_dag_task(end_task, request, False):
        text = output.text
        if text:
            text = text.replace("\n", "\\n")
        if output.error_code != 0:
            yield f"data:[SERVER_ERROR]{text}\n\n"
            break
        else:
            yield f"data:{text}\n\n"


# --- Define request and response models for the new endpoint ---
class SimpleDbChatRequestBody(BaseModel):
    user_input: str = Field(..., description="用户输入的问题")
    db_name: str = Field(..., description="目标数据库名称")
    model_name: Optional[str] = Field(default="zhipu_proxyllm", description="使用的大模型名称")
    conv_uid: Optional[str] = Field(default=None, description="对话ID（可选，不传则新建）")

class SimpleDbChatResponse(BaseModel):
    text_answer: Optional[str] = Field(default="(无文本回答)", description="模型的文本回答或思考过程")
    sql_query: Optional[str] = Field(default=None, description="模型生成的 SQL 查询")
    query_result: Optional[List[Dict]] = Field(default=None, description="SQL 查询的实际结果数据 (如果可用)")


# --- Add the new API route ---
@router.post("/v1/chat/simple_db_chat", response_model=Result[SimpleDbChatResponse])
async def simple_db_chat_completions(
    request: SimpleDbChatRequestBody = Body(),
    user_info: UserRequest = Depends(get_user_from_headers),
):
    """专门为独立前端提供的简化版数据库聊天接口"""
    logger.info(f"simple_db_chat_completions: db_name={request.db_name}, model_name={request.model_name}")
    chat_mode_enum = ChatScene.ChatWithDbExecute
    conv_id = request.conv_uid or str(uuid.uuid4())
    logger.info(f"Using conv_uid: {conv_id}")

    try:
        # Prepare chat_param dictionary
        chat_param = {
            "chat_session_id": conv_id,
            "user_name": user_info.user_name or "anonymous_simple_api",
            "sys_code": getattr(user_info, 'sys_code', None),
            "current_user_input": request.user_input,
            "select_param": request.db_name,
            "model_name": request.model_name,
            "chat_mode": chat_mode_enum,
            "temperature": 0.5,
            "max_new_tokens": 1024,
            "stream": False,
            "app_code": chat_mode_enum.value(),
            'user_id': user_info.user_id
        }
        logger.info(f"Initial chat_param: {chat_param}")

        # Construct ConversationVo
        dialogue = ConversationVo(
            chat_mode=chat_mode_enum.value(),
            conv_uid=conv_id,
            user_input=request.user_input,
            select_param=request.db_name,
            model_name=request.model_name,
            user_name=chat_param["user_name"],
            sys_code=chat_param["sys_code"],
            messages=None,
            app_code=chat_mode_enum.value(),
            temperature=chat_param["temperature"],
            max_new_tokens=chat_param["max_new_tokens"],
            summary=None,
            param_type=None,
            param_value=None,
            ext_info={},
            llm_echo=False,
        )
        logger.info(f"Constructed dialogue object for get_chat_instance: {dialogue}")

        # Get chat instance
        chat_instance: BaseChat = await get_chat_instance(dialogue)
        chat_instance.current_user_input = request.user_input
        logger.info(f"Chat instance created: {type(chat_instance)}, current_user_input set to: {chat_instance.current_user_input}")

        # Call the non-streaming method
        llm_response_str = await chat_instance.nostream_call()
        logger.info(f"Raw nostream_call response string: {llm_response_str}")

        # --- Parse the response string (Revised Robust Approach) ---
        text_answer = llm_response_str # Default to the full string
        sql_query = None
        structured_data = None # To store the parsed JSON from chart-view if found
        query_result_data = None # 新增 query_result_data 变量

        start_tag_str = "<chart-view"
        end_tag_str = "/>"
        content_attr_str = 'content="'

        start_index = llm_response_str.find(start_tag_str)
        end_index = llm_response_str.find(end_tag_str, start_index)

        if start_index != -1 and end_index != -1:
            # Found potential tag bounds
            logger.info("Found potential <chart-view> tag bounds.")
            # Extract the full tag for removal later
            tag_full_string = llm_response_str[start_index : end_index + len(end_tag_str)]
            logger.info(f"Tag full string: {tag_full_string}")

            # Extract content attribute value more carefully
            content_start_index = tag_full_string.find(content_attr_str)
            if content_start_index != -1:
                content_start_index += len(content_attr_str)
                # Find the closing quote for the content attribute
                content_end_index = tag_full_string.find('"', content_start_index)
                if content_end_index != -1:
                    escaped_json = tag_full_string[content_start_index:content_end_index]
                    logger.info(f"Extracted escaped JSON from content attribute: {escaped_json}")

                    # Attempt to decode and parse the extracted JSON
                    try:
                        decoded_json_string = html.unescape(escaped_json) # Use html module
                        logger.info(f"Decoded JSON string: {decoded_json_string}")
                        structured_data = json.loads(decoded_json_string) # Parse the JSON
                        sql_query = structured_data.get('sql') # Extract SQL from parsed JSON
                        logger.info(f"Extracted SQL from structured data: {sql_query}")

                        # Extract query_result_data
                        if structured_data.get('type') == 'response_table' and isinstance(structured_data.get('data'), list):
                            query_result_data = structured_data.get('data')
                            logger.info(f"Extracted table data: {query_result_data}")

                        # Get the text part by removing the tag from the original string
                        text_answer = llm_response_str.replace(tag_full_string, '').strip()
                        logger.info(f"Text part after removing tag: '{text_answer}'")

                        # If text part is empty, try using thoughts from JSON
                        if not text_answer and structured_data:
                             text_answer = structured_data.get('thoughts') or structured_data.get('direct_response') or "(无文本回答)"
                        # Add a fallback if text is still empty after removing tag
                        elif not text_answer:
                             text_answer = "(分析完成，请查看 SQL 和结果)"

                    except Exception as parse_err:
                        logger.error(f"Failed to decode/parse JSON from chart-view content: {parse_err}", exc_info=True)
                        sql_query = "(解析嵌入式 JSON 失败)"
                        text_answer = llm_response_str # Fallback to the original mixed string

                else:
                    logger.warning("Could not find closing quote for content attribute within the tag.")
                    text_answer = llm_response_str # Fallback
                    sql_query = None
            else:
                logger.warning("Could not find 'content=\"' attribute within the tag.")
                text_answer = llm_response_str # Fallback
                sql_query = None
        else:
            # No <chart-view> tag found, attempt to parse the whole string as JSON
            logger.info("No <chart-view> tag found. Attempting to parse entire string as JSON.")
            try:
                structured_data = json.loads(llm_response_str)
                logger.info(f"Successfully parsed entire string as JSON: {structured_data}")
                sql_query = structured_data.get('sql')
                text_answer = structured_data.get('direct_response') or structured_data.get('thoughts') or "(未能提取文本回答)"
            except json.JSONDecodeError:
                 logger.info("Failed to parse entire string as JSON. Treating as plain text.")
                 text_answer = llm_response_str
                 sql_query = None # Plain text has no SQL

        # Ensure text_answer has a fallback value if still empty
        if not text_answer:
             text_answer = "(无有效文本内容)"
        logger.info(f"Text answer BEFORE cleaning: {repr(text_answer)}")

        # --- Return simplified response ---
        # Step 1: Remove actual trailing whitespace
        cleaned_ws = re.sub(r'\s+$', '', text_answer)
        # Step 2: Remove literal "\\n" if it exists at the end
        if cleaned_ws.endswith('\\n'): # Need double backslash in string literal to match one literal backslash
            final_text_answer = cleaned_ws[:-2] # Remove the last two characters
        else:
            final_text_answer = cleaned_ws

        logger.info(f"Text answer AFTER cleaning: {repr(final_text_answer)}")

        response_data = SimpleDbChatResponse(
            text_answer=final_text_answer,
            sql_query=sql_query,
            query_result=query_result_data # Include the extracted data
        )
        logger.info(f"Returning simplified response: {response_data}")
        return Result.succ(response_data)

    except Exception as e:
        logger.error(f"Error in simple_db_chat_completions: {e}", exc_info=True)
        # Return a failure Result object
        return Result.failed(code="E9999", msg=f"处理请求时发生内部错误: {str(e)}")
