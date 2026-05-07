# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import inspect
import json
import logging
import os
import uuid
from pprint import pprint
from typing import Any, Callable, Optional

import numpy as np
import ray
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler.profile import DistProfiler
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, run_unvicorn
from verl.workers.rollout.vllm_rollout import ServerAdapter
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    get_vllm_max_lora_rank,
)

_VLLM_VERSION = version.parse(vllm.__version__)

if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

        get_encoding()
    elif _VLLM_VERSION >= version.parse("0.13.0"):
        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
        """
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = min(
            get_max_position_embeddings(self.model_config.hf_config),
            self.config.prompt_length + self.config.response_length,
        )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)
        self.server_profiler_dir = os.environ.pop("VLLM_TORCH_PROFILER_DIR", None)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        logger.info(
            f"vLLMHttpServer, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization = self.config.quantization

        if quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
                # for subprocesses patching
                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"

        hf_overrides = {}
        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        if quantization == "fp8":
            hf_overrides["quantization_config"] = fp8_block_quant_kwargs

        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            speculative_config = {
                "method": self.config.mtp.method,
                "num_speculative_tokens": self.config.mtp.num_speculative_tokens,
            }
            args["speculative_config"] = speculative_config

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        megatron_lora = True
        if self.model_config.lora.get("merge", False):
            lora_rank = 0
        if lora_rank <= 0:
            megatron_lora = False
            lora_rank = self.model_config.lora_rank
        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if megatron_lora:
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            self._master_sock.close()
            await self.run_server(server_args)
        else:
            # TODO: avoid connect before master_sock close
            await asyncio.sleep(3)
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        app = build_app(args)
        if _VLLM_VERSION > version.parse("0.11.0"):
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # Clamp max_tokens to the valid range [0, max_possible_tokens]
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # Add lora request
        lora_request = None
        if self.model_config.lora_rank > 0 or (
            self.model_config.lora.get("rank", 0) > 0 and not self.model_config.lora.get("merge", False)
        ):
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
        )

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            if self.node_rank == 0:
                await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
                await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
            and self.server_profiler_dir
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
            and self.server_profiler_dir
        ):
            await self.engine.stop_profile()

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            # For each request, create an abort output and put it to its queue
            # This allows the generator to receive the aborted result
            from vllm.v1.engine import FinishReason

            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            # Abort requests in the output processor and engine core
            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    async def _fepo_raw_generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        logprobs_k: int | None,
    ) -> RequestOutput:
        """Single vLLM async generate returning full ``RequestOutput`` (for FEPO MC / teacher forcing)."""
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )
        max_tokens = max(0, min(int(max_tokens), max_possible_tokens))
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            logprobs=logprobs_k,
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
        )
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data={})

        lora_request = None
        if self.model_config.lora_rank > 0 or (
            self.model_config.lora.get("rank", 0) > 0 and not self.model_config.lora.get("merge", False)
        ):
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sp,
            request_id=uuid.uuid4().hex,
            lora_request=lora_request,
            priority=0,
        )
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
        return final_res

    async def _fepo_mc_mean_f_per_prefix(
        self,
        prefixes: list[list[int]],
        *,
        m_samples: int,
        gen_cap: int,
        mc_temperature: float,
        mc_top_p: float,
        k_lp: int,
        f_mode: str,
        norm_len: bool,
        tokenizer: Any,
        stop_fn: Any,
        batch_chunk: int,
    ) -> list[float]:
        """Per prefix: MC mean of continuation-F (same definition as ``entropy_credit_experiment``)."""
        from verl.utils.fepo_math import continuation_F_from_gen_ids_and_step_logprobs, vllm_request_step_logprobs_to_float_dicts

        if not prefixes:
            return []
        m_samples = max(1, int(m_samples))
        flat_prompts: list[list[int]] = []
        flat_pref_idx: list[int] = []
        for pi, pref in enumerate(prefixes):
            for _ in range(m_samples):
                flat_prompts.append(list(pref))
                flat_pref_idx.append(pi)
        n_total = len(flat_prompts)
        per_pref: list[list[float]] = [[] for _ in range(len(prefixes))]
        bc = max(1, int(batch_chunk))

        async def _one_mc_sample(pref_ids: list[int], pi: int) -> tuple[int, float]:
            out = await self._fepo_raw_generate(
                pref_ids,
                max_tokens=int(gen_cap),
                temperature=float(mc_temperature),
                top_p=float(mc_top_p),
                logprobs_k=int(k_lp),
            )
            o = out.outputs[0]
            gen_ids = list(o.token_ids)
            step_lps = vllm_request_step_logprobs_to_float_dicts(o)
            while len(step_lps) < len(gen_ids):
                step_lps.append({})
            val = continuation_F_from_gen_ids_and_step_logprobs(
                gen_ids,
                step_lps,
                f_continuation_mode=f_mode,
                tokenizer=tokenizer if f_mode == "first_sentence" else None,
                stop_fn=stop_fn if f_mode == "first_sentence" else None,
                normalize_by_continuation_length=norm_len,
            )
            return pi, float(val)

        for start in range(0, n_total, bc):
            end = min(start + bc, n_total)
            chunk_results = await asyncio.gather(
                *[_one_mc_sample(flat_prompts[j], flat_pref_idx[j]) for j in range(start, end)]
            )
            for pi, val in chunk_results:
                per_pref[pi].append(val)
        return [float(np.mean(xs)) if xs else 0.0 for xs in per_pref]

    async def _fepo_mc_mean_future_rate_minus_ht(
        self,
        prefixes: list[list[int]],
        *,
        m_samples: int,
        gen_cap: int,
        mc_temperature: float,
        mc_top_p: float,
        k_lp: int,
        f_mode: str,
        tokenizer: Any,
        stop_fn: Any,
        batch_chunk: int,
        h_t_values: list[float],
    ) -> list[float]:
        """Per prefix: MC mean of ``(sum_H - H_t) / (len-1)`` after continuation truncation."""
        from verl.utils.fepo_math import entropy_from_logprobs_topk, vllm_request_step_logprobs_to_float_dicts
        from verl.utils.fepo_sentence_stop import truncate_gen_ids_to_first_sentence

        if not prefixes:
            return []
        m_samples = max(1, int(m_samples))
        flat_prompts: list[list[int]] = []
        flat_pref_idx: list[int] = []
        for pi, pref in enumerate(prefixes):
            for _ in range(m_samples):
                flat_prompts.append(list(pref))
                flat_pref_idx.append(pi)
        n_total = len(flat_prompts)
        per_pref: list[list[float]] = [[] for _ in range(len(prefixes))]
        bc = max(1, int(batch_chunk))

        async def _one_mc_sample(pref_ids: list[int], pi: int) -> tuple[int, float]:
            out = await self._fepo_raw_generate(
                pref_ids,
                max_tokens=int(gen_cap),
                temperature=float(mc_temperature),
                top_p=float(mc_top_p),
                logprobs_k=int(k_lp),
            )
            o = out.outputs[0]
            gen_ids = list(o.token_ids)
            step_lps = vllm_request_step_logprobs_to_float_dicts(o)
            while len(step_lps) < len(gen_ids):
                step_lps.append({})

            if f_mode == "first_sentence":
                keep_k = truncate_gen_ids_to_first_sentence(gen_ids, tokenizer, stop_fn)
            else:
                keep_k = len(gen_ids)
            keep_k = max(0, int(keep_k))
            if keep_k <= 1:
                return pi, 0.0

            s = 0.0
            for i in range(keep_k):
                s += float(entropy_from_logprobs_topk(step_lps[i]))
            h_t = float(h_t_values[pi])
            val = (s - h_t) / float(max(keep_k - 1, 1))
            return pi, float(val)

        for start in range(0, n_total, bc):
            end = min(start + bc, n_total)
            chunk_results = await asyncio.gather(
                *[_one_mc_sample(flat_prompts[j], flat_pref_idx[j]) for j in range(start, end)]
            )
            for pi, val in chunk_results:
                per_pref[pi].append(val)
        return [float(np.mean(xs)) if xs else 0.0 for xs in per_pref]

    async def _fepo_teacher_forced_f_real(
        self,
        *,
        prefix_after_chosen: list[int],
        suffix_after: list[int],
        k_lp: int,
        f_mode: str,
        norm_len: bool,
        tokenizer: Any,
        stop_fn: Any,
    ) -> float:
        """Teacher-forced f_real on the actual rollout suffix (same continuation F/truncation)."""
        from verl.utils.fepo_math import continuation_F_from_gen_ids_and_step_logprobs, vllm_request_step_logprobs_to_float_dicts

        if not suffix_after:
            return 0.0
        cur = list(prefix_after_chosen)
        step_lps_tf: list[dict[int, float]] = []
        for tok in suffix_after:
            out_tf = await self._fepo_raw_generate(
                cur,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs_k=int(k_lp),
            )
            o0 = out_tf.outputs[0]
            per_step = vllm_request_step_logprobs_to_float_dicts(o0)
            if per_step:
                step_lps_tf.append(per_step[0])
            else:
                step_lps_tf.append({})
            cur.append(int(tok))
        return float(
            continuation_F_from_gen_ids_and_step_logprobs(
                list(suffix_after),
                step_lps_tf,
                f_continuation_mode=f_mode,
                tokenizer=tokenizer if f_mode == "first_sentence" else None,
                stop_fn=stop_fn if f_mode == "first_sentence" else None,
                normalize_by_continuation_length=norm_len,
            )
        )

    async def _fepo_probe_candidates_batch(
        self,
        prefixes: list[list[int]],
        *,
        k_lp: int,
        candidate_top_p: float,
        candidate_max_k: int,
        batch_chunk: int,
    ) -> list[tuple[list[int], list[float]]]:
        """Batch probe one-step next-token candidates for many FEPO prefixes."""
        from verl.utils.fepo_branching import topp_capped_candidates_from_step_logprobs

        if not prefixes:
            return []

        bc = max(1, int(batch_chunk))
        out: list[tuple[list[int], list[float]]] = [([], []) for _ in range(len(prefixes))]

        async def _probe_one(pref_ids: list[int]) -> tuple[list[int], list[float]]:
            out0 = await self._fepo_raw_generate(
                pref_ids,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs_k=k_lp,
            )
            o0 = out0.outputs[0]
            step_lp: dict[int, float] = {}
            if o0.logprobs and len(o0.logprobs) > 0:
                for tid, info in o0.logprobs[0].items():
                    step_lp[int(tid)] = float(info.logprob)
            cands, cand_probs = topp_capped_candidates_from_step_logprobs(step_lp, candidate_top_p, candidate_max_k)
            return cands, cand_probs

        for start in range(0, len(prefixes), bc):
            end = min(start + bc, len(prefixes))
            batch_results = await asyncio.gather(*[_probe_one(prefixes[i]) for i in range(start, end)])
            for local_i, item in enumerate(batch_results):
                out[start + local_i] = item
        return out

    async def fepo_compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """FEPO branching: ``f_bar = E_c[ F_MC(prefix+c) ]``, ``f_real = F_MC(prefix+chosen)`` (``compare_bias``)."""
        from verl.utils.fepo_math import clamp_vllm_logprobs_topk
        from verl.utils.fepo_sentence_stop import completion_should_stop_after_first_sentence_simple

        tokenizer = self.model_config.tokenizer
        jobs = payload["jobs"]
        mc_m = max(1, int(payload.get("mc_m", 1)))
        mc_temperature = float(payload.get("mc_temperature", 1.0))
        mc_top_p = float(payload.get("mc_top_p", 0.95))
        k_lp = clamp_vllm_logprobs_topk(int(payload.get("logprobs_k", 20)))
        f_mode = str(payload.get("f_continuation_mode", "first_sentence"))
        norm_len = bool(payload.get("normalize_by_continuation_length", True))
        candidate_top_p = float(payload.get("candidate_top_p", 0.95))
        candidate_max_k = int(payload.get("candidate_max_k", 20))
        candidate_min_prob = float(payload.get("candidate_min_prob", 0.0))
        min_candidates = max(2, int(payload.get("min_candidates", 2)))
        mc_batch_chunk = max(1, int(payload.get("mc_batch_chunk", 32)))
        probe_batch_chunk = max(1, int(payload.get("probe_batch_chunk", mc_batch_chunk)))
        f_bar_mode = str(payload.get("f_bar_mode", "branching"))
        f_real_mode = str(payload.get("f_real_mode", "chosen_branch_mc"))
        global_pooling = bool(payload.get("global_pooling", False))
        detail_full = bool(payload.get("detail_full", False))
        if f_bar_mode not in {"branching", "prefix_minus_ht"}:
            f_bar_mode = "branching"
        if f_real_mode not in {"chosen_branch_mc", "teacher_forced_real_path"}:
            f_real_mode = "chosen_branch_mc"

        stop_fn = completion_should_stop_after_first_sentence_simple if f_mode == "first_sentence" else None

        n_jobs = len(jobs)
        deltas: list[float] = [0.0] * n_jobs
        ok_flags: list[bool] = [False] * n_jobs
        details: list[dict[str, Any]] = [{} for _ in range(n_jobs)]

        need_cands = (f_bar_mode == "branching") or (f_real_mode == "chosen_branch_mc")
        runnable_idx: list[int] = []
        prefix_for_probe: list[list[int]] = []
        probe_owner_idx: list[int] = []
        cands_by_job: list[list[int]] = [[] for _ in range(n_jobs)]
        cand_probs_by_job: list[list[float]] = [[] for _ in range(n_jobs)]

        for jidx, job in enumerate(jobs):
            suffix_after = list(job.get("suffix_after", []))
            if not suffix_after:
                details[jidx] = {"reason": "empty_suffix", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                continue
            runnable_idx.append(jidx)
            if need_cands:
                prefix_for_probe.append(list(job["prefix_before"]))
                probe_owner_idx.append(jidx)

        if need_cands and prefix_for_probe:
            cand_results = await self._fepo_probe_candidates_batch(
                prefix_for_probe,
                k_lp=k_lp,
                candidate_top_p=candidate_top_p,
                candidate_max_k=candidate_max_k,
                batch_chunk=probe_batch_chunk,
            )
            for local_i, (cands, cand_probs) in enumerate(cand_results):
                jidx = probe_owner_idx[local_i]
                # Drop low-probability candidates to reduce MC cost, but always keep ``chosen``
                # if it is present so that f_real for chosen_branch_mc stays available.
                if candidate_min_prob > 0.0 and cands:
                    chosen_tok = int(jobs[jidx]["chosen_token"])
                    kept: list[int] = []
                    kept_p: list[float] = []
                    for c, p in zip(cands, cand_probs, strict=True):
                        if float(p) >= candidate_min_prob or int(c) == chosen_tok:
                            kept.append(int(c))
                            kept_p.append(float(p))
                    if kept:
                        s = float(sum(kept_p))
                        if s > 0.0 and np.isfinite(s):
                            kept_p = [float(x) / s for x in kept_p]
                        cands = kept
                        cand_probs = kept_p
                cands_by_job[jidx] = cands
                cand_probs_by_job[jidx] = cand_probs

        compute_idx: list[int] = []
        for jidx in runnable_idx:
            job = jobs[jidx]
            cands = cands_by_job[jidx]
            cand_probs = cand_probs_by_job[jidx]
            chosen = int(job["chosen_token"])

            if need_cands and len(cands) < min_candidates:
                d = {"reason": "insufficient_candidates", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                if detail_full:
                    d["cands"] = cands
                    d["cand_probs"] = cand_probs
                details[jidx] = d
                continue
            # Early reject before expensive MC when chosen branch is unavailable.
            if f_real_mode == "chosen_branch_mc" and chosen not in cands:
                d = {"reason": "chosen_not_in_candidates", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                if detail_full:
                    d["cands"] = cands
                    d["cand_probs"] = cand_probs
                details[jidx] = d
                continue
            compute_idx.append(jidx)

        f_mc_by_job: list[list[float] | None] = [None for _ in range(n_jobs)]
        # Optional global pooling: aggregate branch MC requests across jobs, then scatter.
        if global_pooling and need_cands and compute_idx:
            gen_cap_groups: dict[int, list[int]] = {}
            for jidx in compute_idx:
                gc = max(1, int(jobs[jidx]["cont_max_new_tokens"]))
                gen_cap_groups.setdefault(gc, []).append(jidx)

            next_compute_idx: list[int] = []
            for gc, gidx in gen_cap_groups.items():
                pooled_prefixes: list[list[int]] = []
                pooled_owner_and_len: list[tuple[int, int, int]] = []
                for jidx in gidx:
                    job = jobs[jidx]
                    pb = list(job["prefix_before"])
                    cands = cands_by_job[jidx]
                    start = len(pooled_prefixes)
                    pooled_prefixes.extend([pb + [int(c)] for c in cands])
                    pooled_owner_and_len.append((jidx, start, len(cands)))

                if not pooled_prefixes:
                    continue

                pooled_vals = await self._fepo_mc_mean_f_per_prefix(
                    pooled_prefixes,
                    m_samples=mc_m,
                    gen_cap=int(gc),
                    mc_temperature=mc_temperature,
                    mc_top_p=mc_top_p,
                    k_lp=k_lp,
                    f_mode=f_mode,
                    norm_len=norm_len,
                    tokenizer=tokenizer,
                    stop_fn=stop_fn,
                    batch_chunk=int(mc_batch_chunk),
                )
                for jidx, start, cnt in pooled_owner_and_len:
                    sub = pooled_vals[start : start + cnt]
                    if len(sub) != cnt:
                        d = {"reason": "mc_size_mismatch", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                        if detail_full:
                            d["cands"] = cands_by_job[jidx]
                            d["cand_probs"] = cand_probs_by_job[jidx]
                            d["f_mc"] = sub
                        details[jidx] = d
                        continue
                    f_mc_by_job[jidx] = [float(x) for x in sub]
                    next_compute_idx.append(jidx)
            compute_idx = next_compute_idx

        # Optional pooling for prefix_minus_ht f_bar (grouped by gen_cap).
        f_bar_pmht_by_job: list[float | None] = [None for _ in range(n_jobs)]
        if global_pooling and f_bar_mode == "prefix_minus_ht" and compute_idx:
            gen_cap_groups: dict[int, list[int]] = {}
            for jidx in compute_idx:
                gc = max(1, int(jobs[jidx]["cont_max_new_tokens"]) + 1)
                gen_cap_groups.setdefault(gc, []).append(jidx)
            for gc, gidx in gen_cap_groups.items():
                prefixes = [list(jobs[j]["prefix_before"]) for j in gidx]
                hvals = [float(jobs[j].get("h_t", 0.0)) for j in gidx]
                vals = await self._fepo_mc_mean_future_rate_minus_ht(
                    prefixes,
                    m_samples=mc_m,
                    gen_cap=int(gc),
                    mc_temperature=mc_temperature,
                    mc_top_p=mc_top_p,
                    k_lp=k_lp,
                    f_mode=f_mode,
                    tokenizer=tokenizer,
                    stop_fn=stop_fn,
                    batch_chunk=int(mc_batch_chunk),
                    h_t_values=hvals,
                )
                for local_i, jidx in enumerate(gidx):
                    if local_i < len(vals):
                        f_bar_pmht_by_job[jidx] = float(vals[local_i])

        job_concurrency = max(1, min(int(payload.get("fepo_job_concurrency", 8)), len(compute_idx) if compute_idx else 1))

        async def _compute_one_job(jidx: int) -> tuple[int, float, bool, dict[str, Any]]:
            job = jobs[jidx]
            pb = list(job["prefix_before"])
            chosen = int(job["chosen_token"])
            gen_cap = max(1, int(job["cont_max_new_tokens"]))
            suffix_after = list(job.get("suffix_after", []))
            cands = cands_by_job[jidx]
            cand_probs = cand_probs_by_job[jidx]
            job_batch_chunk = int(mc_batch_chunk)
            f_mc: list[float] = []
            if need_cands:
                if global_pooling:
                    cached = f_mc_by_job[jidx]
                    if cached is None:
                        d = {"reason": "mc_not_precomputed", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                        if detail_full:
                            d["cands"] = cands
                            d["cand_probs"] = cand_probs
                        return (jidx, 0.0, False, d)
                    f_mc = cached
                else:
                    prefixes = [pb + [int(c)] for c in cands]
                    f_mc = await self._fepo_mc_mean_f_per_prefix(
                        prefixes,
                        m_samples=mc_m,
                        gen_cap=gen_cap,
                        mc_temperature=mc_temperature,
                        mc_top_p=mc_top_p,
                        k_lp=k_lp,
                        f_mode=f_mode,
                        norm_len=norm_len,
                        tokenizer=tokenizer,
                        stop_fn=stop_fn,
                        batch_chunk=job_batch_chunk,
                    )
                    if len(f_mc) != len(cands):
                        d = {"reason": "mc_size_mismatch", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode}
                        if detail_full:
                            d["cands"] = cands
                            d["cand_probs"] = cand_probs
                            d["f_mc"] = f_mc
                        return (jidx, 0.0, False, d)

            if f_bar_mode == "prefix_minus_ht":
                if global_pooling:
                    cached_bar = f_bar_pmht_by_job[jidx]
                    f_bar = float(cached_bar) if cached_bar is not None else 0.0
                else:
                    f_bar_list = await self._fepo_mc_mean_future_rate_minus_ht(
                        [pb],
                        m_samples=mc_m,
                        gen_cap=max(1, gen_cap + 1),
                        mc_temperature=mc_temperature,
                        mc_top_p=mc_top_p,
                        k_lp=k_lp,
                        f_mode=f_mode,
                        tokenizer=tokenizer,
                        stop_fn=stop_fn,
                        batch_chunk=job_batch_chunk,
                        h_t_values=[float(job.get("h_t", 0.0))],
                    )
                    f_bar = float(f_bar_list[0]) if f_bar_list else 0.0
            else:
                if not cands:
                    return (
                        jidx,
                        0.0,
                        False,
                        {"reason": "no_candidates", "f_bar_mode": f_bar_mode, "f_real_mode": f_real_mode},
                    )
                f_bar = float(sum(float(p) * float(f_mc[i]) for i, p in enumerate(cand_probs)))

            if f_real_mode == "teacher_forced_real_path":
                f_real = await self._fepo_teacher_forced_f_real(
                    prefix_after_chosen=pb + [chosen],
                    suffix_after=suffix_after[:gen_cap],
                    k_lp=k_lp,
                    f_mode=f_mode,
                    norm_len=norm_len,
                    tokenizer=tokenizer,
                    stop_fn=stop_fn,
                )
            else:
                f_real = float(f_mc[cands.index(chosen)])

            detail: dict[str, Any] = {
                "f_bar_mode": f_bar_mode,
                "f_real_mode": f_real_mode,
                "f_bar": float(f_bar),
                "f_real": float(f_real),
                "branch_min_f_mc": float(min(f_mc)) if f_mc else None,
                "mc_batch_chunk_used": int(mc_batch_chunk),
            }
            if detail_full:
                detail["cands"] = [int(x) for x in cands] if cands else []
                detail["cand_probs"] = [float(x) for x in cand_probs] if cand_probs else []
                detail["f_mc"] = [float(x) for x in f_mc] if f_mc else []

            return (jidx, float(f_bar - f_real), True, detail)

        for start in range(0, len(compute_idx), job_concurrency):
            end = min(start + job_concurrency, len(compute_idx))
            chunk_res = await asyncio.gather(*[_compute_one_job(jidx) for jidx in compute_idx[start:end]])
            for jidx, delta_v, ok_v, detail_v in chunk_res:
                deltas[jidx] = float(delta_v)
                ok_flags[jidx] = bool(ok_v)
                details[jidx] = detail_v

        return {"deltas": deltas, "ok": ok_flags, "details": details}


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # NOTE: We always use MP Executor backend whether it's single-node or multi-node.
        # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
        if self.config.data_parallel_size == 1 and self.nnodes > 1:
            assert _VLLM_VERSION >= version.parse("0.11.1"), (
                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
            )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
