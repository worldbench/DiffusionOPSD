# Compatibility shims for the validated Transformers 4.51 stack and Z-Image tokenizer loading.
# They run before scorer or pipeline modules load and are no-ops whenever the required symbols and
# chat template are already available.
def _apply_tfmr_compat():
    try:
        import transformers.image_utils as _iu, transformers.modeling_utils as _mu, transformers.pytorch_utils as _pu
        if not hasattr(_iu, "VideoInput"):
            try:
                from transformers.video_utils import VideoInput as _VI
                _iu.VideoInput = _VI
            except Exception:
                _iu.VideoInput = list
        for _n in dir(_pu):
            if not _n.startswith("__") and not hasattr(_mu, _n):
                setattr(_mu, _n, getattr(_pu, _n))
    except Exception:
        pass


def _patch_chat_template_autoload():
    # ZImagePipeline loads its Qwen tokenizer via the diffusers subfolder mechanism, under which tfmr
    # 4.51 does NOT populate tokenizer.chat_template (direct-from-dir load DOES). Lazily load it from the
    # tokenizer_config.json under name_or_path OR name_or_path/tokenizer. No-op when already set (4.57).
    try:
        import os, json
        from transformers import PreTrainedTokenizerBase
        _orig = PreTrainedTokenizerBase.get_chat_template
        def get_chat_template(self, chat_template=None, tools=None):
            if chat_template is None and getattr(self, "chat_template", None) is None:
                nop = getattr(self, "name_or_path", None)
                if nop:
                    for cfg in (os.path.join(str(nop), "tokenizer_config.json"),
                                os.path.join(str(nop), "tokenizer", "tokenizer_config.json")):
                        if os.path.isfile(cfg):
                            try:
                                ct = json.load(open(cfg)).get("chat_template")
                                if ct:
                                    self.chat_template = ct
                                    break
                            except Exception:
                                pass
            return _orig(self, chat_template, tools)
        PreTrainedTokenizerBase.get_chat_template = get_chat_template
    except Exception:
        pass


_apply_tfmr_compat()
_patch_chat_template_autoload()
