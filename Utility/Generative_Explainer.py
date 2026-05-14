"""LLM-based generative explainer for XG-NID predictions (block #6 of paper).

Builds a structured prompt from the IG explanation + raw flow/packet context
following the paper's Algorithm-3 spirit:
  1. Header: declared attack class + confidence
  2. Flow context: top-k flow features that drove the decision (name, value,
     IG attribution sign/magnitude)
  3. Payload context: top-k payload byte indices + their values + decoded
     ASCII fragment of the first packet's payload
  4. Instruction: ask the LLM to explain why this is the predicted attack and
     identify attack signatures + recommended mitigation

The LLM backend is pluggable. Default is a "preview" backend that just prints
the prompt so the pipeline runs without heavy weights. To use a real model,
pass an instance of HFCausalBackend (transformers + bitsandbytes 4-bit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .Explainer import IGExplanation, FLOW_FEATURE_NAMES_82


CLASS_NAMES_8 = ["Benign", "WebBased", "Spoofing", "Recon",
                 "Mirai", "Dos", "DDos", "BruteForce"]


@dataclass
class GenerativeExplanation:
    prompt: str
    response: str


def _decode_payload_ascii(payload_bytes_uint8: np.ndarray, max_chars: int = 200) -> str:
    """Render printable ASCII chars from a byte vector; replace others with '.'."""
    out = []
    for b in payload_bytes_uint8.tolist()[:max_chars]:
        b = int(b) & 0xFF
        if 0x20 <= b < 0x7f:
            out.append(chr(b))
        elif b == 0x0a:
            out.append("\\n")
        elif b == 0x0d:
            out.append("\\r")
        elif b == 0x09:
            out.append("\\t")
        else:
            out.append(".")
    return "".join(out)


def build_prompt(
    explanation: IGExplanation,
    flow_x: np.ndarray,           # (num_flow_nodes, 82)
    packet_x: np.ndarray,         # (num_packet_nodes, 1500)
    class_names: List[str] = CLASS_NAMES_8,
) -> str:
    cls = class_names[explanation.pred_class]
    flow_vec = flow_x[0]

    # 3a) Flow context
    flow_lines = []
    for name, attr in explanation.top_flow:
        idx = FLOW_FEATURE_NAMES_82.index(name)
        val = float(flow_vec[idx])
        sign = "+" if attr >= 0 else "-"
        flow_lines.append(f"  - {name} = {val:g}  (IG {sign}{abs(attr):.3e})")

    # 3b) Payload context: pick first packet, render ASCII, and list top bytes
    first_payload = packet_x[0] if len(packet_x) else np.zeros(1500, dtype=np.uint8)
    ascii_render = _decode_payload_ascii(first_payload.astype(np.uint8))
    byte_lines = []
    for idx, attr in explanation.top_payload_bytes:
        val = int(first_payload[idx]) if idx < len(first_payload) else 0
        sign = "+" if attr >= 0 else "-"
        byte_lines.append(f"  - byte[{idx}] = 0x{val:02x} ({val})  (IG {sign}{abs(attr):.3e})")

    parts = [
        "You are a network security analyst reviewing a flow flagged by the XG-NID HGNN classifier.",
        f"Predicted class: {cls} (log-prob={explanation.pred_logprob:.3f}).",
        "",
        "Top flow-level features driving the decision (name = value, with Integrated Gradients attribution):",
        *flow_lines,
        "",
        "Top payload bytes driving the decision (byte index, value, IG attribution):",
        *byte_lines,
        "",
        "First packet payload (printable ASCII rendering, first 200 bytes):",
        f"  '{ascii_render}'",
        "",
        "Tasks:",
        f"1. Explain in 3-5 sentences why this flow is consistent with {cls}, citing the specific flow features above.",
        "2. If the payload appears to contain a known attack signature (HTTP exploit, SQLi, XSS, command injection, scanner banner, brute-force protocol, malware control plane, etc.), name it and quote the relevant byte range.",
        "3. Recommend 2 concrete mitigations a defender should apply.",
        "Write a concise expert-level analysis. Do not invent features that were not provided.",
    ]
    return "\n".join(parts)


class PreviewBackend:
    """No-LLM backend: returns the prompt itself. Useful for tests / no-GPU runs."""

    def generate(self, prompt: str) -> str:
        return ("[PreviewBackend: no LLM invoked. Returning prompt as response.]\n\n"
                + prompt)


class HFCausalBackend:
    """HuggingFace causal LM backend.

    Defaults to a model that does not require a gated HF access token. The
    paper used Llama-3-8B; substitute via model_id if you have access.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 400,
                 load_in_4bit: bool = False,
                 device_map: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, **kwargs)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        msgs = [
            {"role": "system",
             "content": "You are an expert network security analyst."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def explain_generative(
    explanation: IGExplanation,
    flow_x: np.ndarray,
    packet_x: np.ndarray,
    backend: Optional[object] = None,
) -> GenerativeExplanation:
    if backend is None:
        backend = PreviewBackend()
    prompt = build_prompt(explanation, flow_x, packet_x)
    response = backend.generate(prompt)
    return GenerativeExplanation(prompt=prompt, response=response)
