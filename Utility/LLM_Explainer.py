"""
Generative Explainer for XG-NID, paper Section 3.1.6 / Algorithm 3.

Builds a structured zero-shot prompt from the IG attributions produced by
Utility.IG_Explainer and queries a Llama-3-8B model. For payload-specific
attack classes (WebBased, BruteForce in CIC-IoT2023), an additional payload
query is sent and the two responses are concatenated, mirroring G_exp =
R_flow + R_payload in Algorithm 3.

Loading the LLM is optional: pass `pipeline=None` to `LLMExplainer.explain`
to get back the formatted prompts only -- useful for unit tests and for
running the explainer without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .IG_Explainer import IGAttribution, top_flow_features, top_payload_bytes


# Default class -> name table for CIC-IoT2023, paper Tables 2 / 4.
DEFAULT_CLASS_NAMES = {
    0: 'Benign', 1: 'WebBased', 2: 'Spoofing', 3: 'Recon',
    4: 'Mirai', 5: 'DoS', 6: 'DDoS', 7: 'BruteForce',
}

# Classes whose maliciousness is encoded in the payload (Algorithm 3 line 18).
PAYLOAD_SPECIFIC_CLASSES = {'WebBased', 'BruteForce'}


# ---------------------------------------------------------------------------
# Prompt templates  (verbatim from paper Section 3.1.6)
# ---------------------------------------------------------------------------

P_INIT_TEMPLATE = "The predicted class from GNN is {predicted_class}."

P_PART2_HEADER = "The top features contributing to this prediction are:"

P_ALIGN = (
    "Don't expect any values on your own. Explain the predicted outcome and "
    "its potential reason along with the potential mitigation. Start your "
    'answer with "The predicted outcome is."'
)

P_PAYLOAD_PREFIX = (
    "Analyze whether this payload of network flow is malicious or not. "
    "Give reason concisely."
)


@dataclass
class GenerativeExplanation:
    flow_prompt: str
    flow_response: Optional[str]
    payload_prompt: Optional[str]
    payload_response: Optional[str]

    @property
    def text(self) -> str:
        """Combined human-readable explanation, G_exp in Algorithm 3."""
        parts = []
        if self.flow_response is not None:
            parts.append(self.flow_response)
        if self.payload_response is not None:
            parts.append(self.payload_response)
        return "\n\n".join(parts)


class LLMExplainer:
    """Composes prompts from IG attributions and (optionally) calls Llama-3."""

    def __init__(self, class_names: Optional[dict] = None,
                 payload_specific_classes: Optional[set] = None,
                 top_n_features: int = 5, top_n_payload_bytes: int = 64):
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.payload_specific_classes = (
            payload_specific_classes if payload_specific_classes is not None
            else PAYLOAD_SPECIFIC_CLASSES
        )
        self.top_n_features = top_n_features
        self.top_n_payload_bytes = top_n_payload_bytes

    # ------------------------------------------------------------------
    def explain(self, attribution: IGAttribution,
                feature_names: Sequence[str],
                pipeline=None) -> GenerativeExplanation:
        """Build the prompts and (if `pipeline` is provided) call the LLM.

        `pipeline` is expected to be a HuggingFace text-generation pipeline
        wrapping a Llama-3-8B (or compatible) instruction-tuned model. Pass
        None for a dry run that just returns the prompts.
        """
        flow_prompt = self._build_flow_prompt(attribution, feature_names)
        flow_response = self._call_llm(pipeline, flow_prompt)

        payload_prompt = None
        payload_response = None
        class_name = self.class_names.get(attribution.predicted_class,
                                          str(attribution.predicted_class))
        if class_name in self.payload_specific_classes:
            payload_prompt = self._build_payload_prompt(attribution)
            payload_response = self._call_llm(pipeline, payload_prompt)

        return GenerativeExplanation(
            flow_prompt=flow_prompt,
            flow_response=flow_response,
            payload_prompt=payload_prompt,
            payload_response=payload_response,
        )

    # ------------------------------------------------------------------
    def _build_flow_prompt(self, attribution: IGAttribution,
                           feature_names: Sequence[str]) -> str:
        class_name = self.class_names.get(attribution.predicted_class,
                                          str(attribution.predicted_class))
        p_init = P_INIT_TEMPLATE.format(predicted_class=class_name)

        ranked = top_flow_features(attribution, feature_names,
                                   top_n=self.top_n_features)
        feature_lines = [
            f"- {name} with actual value {value:.6g}"
            for name, _attr, value in ranked
        ]
        p_part2 = P_PART2_HEADER + "\n" + "\n".join(feature_lines)

        return f"{p_init}\n\n{p_part2}\n\n{P_ALIGN}"

    def _build_payload_prompt(self, attribution: IGAttribution) -> str:
        # Algorithm 3 lines 19-24: normalize, average, take top-N, convert to
        # ASCII string.  Non-printable bytes are kept as their hex escape.
        top_bytes = top_payload_bytes(attribution, top_n=self.top_n_payload_bytes)
        ascii_repr = ''.join(
            chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in top_bytes
        )
        return f"{P_PAYLOAD_PREFIX}\n\nPayload (top contributing bytes): {ascii_repr}\n\n{P_ALIGN}"

    # ------------------------------------------------------------------
    def _call_llm(self, pipeline, prompt: str) -> Optional[str]:
        if pipeline is None:
            return None
        # HuggingFace text-generation pipeline returns a list of dicts.
        out = pipeline(prompt, max_new_tokens=512, do_sample=False)
        if isinstance(out, list) and out:
            text = out[0].get('generated_text', '')
            # Strip the echoed prompt when the pipeline includes it.
            if text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            return text
        return None


# ---------------------------------------------------------------------------
# Optional helper to load the Llama-3-8B pipeline. Kept separate so that
# importing this module never triggers a model download.
# ---------------------------------------------------------------------------

def load_llama3_pipeline(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                        device_map: str = "auto",
                        load_in_4bit: bool = True):
    """Lazy import + load. Requires `transformers`, `accelerate`, and
    optionally `bitsandbytes` for 4-bit quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
