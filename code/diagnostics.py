"""Optional training-time diagnostics, kept separate from the trainer."""
import json
import math

import torch


class TrainingDiagnostics:
    """Collect sampled gradient and residual-update metrics for a model."""

    def __init__(self, model):
        self.blocks = getattr(model, 'blocks', ())
        self.rows = []
        self._enabled = False
        self._activations = {}
        self._handles = []
        for index, block in enumerate(self.blocks):
            if all(hasattr(block, name) for name in ('proj', 'norm2', 'mlp')):
                self._handles.extend((
                    # --- Added activation RMS capture points ---
                    block.register_forward_pre_hook(self._capture_input(index, 'block_input')),
                    block.register_forward_hook(self._capture_output(index, 'after_ffn_residual')),
                    block.norm2.register_forward_pre_hook(
                        self._capture_input(index, 'after_attention_residual')),
                    # --- End added activation RMS capture points ---
                    block.register_forward_pre_hook(self._capture_input(index, 'attn_base')),
                    block.proj.register_forward_hook(self._capture_output(index, 'attn_delta')),
                    block.norm2.register_forward_pre_hook(self._capture_input(index, 'ffn_base')),
                    block.mlp.register_forward_hook(self._capture_output(index, 'ffn_delta')),
                ))

    def _capture_input(self, index, key):
        def hook(_module, inputs):
            if self._enabled:
                self._activations.setdefault(index, {})[key] = inputs[0].detach()
        return hook

    def _capture_output(self, index, key):
        def hook(_module, _inputs, output):
            if self._enabled:
                self._activations.setdefault(index, {})[key] = output.detach()
        return hook

    def begin_step(self, enabled):
        self._enabled = enabled
        self._activations.clear()

    def residual_update_ratios(self):
        ratios_by_block = {}
        for index, values in self._activations.items():
            ratios = {}
            for branch, base_key, delta_key in (
                ('attention', 'attn_base', 'attn_delta'),
                ('ffn', 'ffn_base', 'ffn_delta'),
            ):
                if base_key in values and delta_key in values:
                    base_norm = torch.linalg.vector_norm(values[base_key].float())
                    delta_norm = torch.linalg.vector_norm(values[delta_key].float())
                    ratios[branch] = (delta_norm / base_norm.clamp_min(1e-12)).item()
            ratios_by_block[f'block_{index}'] = ratios
        self._enabled = False
        self._activations.clear()
        return ratios_by_block

    # --- Added per-block activation RMS ---
    def activation_rms(self):
        rms_by_block = {}
        for index, values in self._activations.items():
            rms_by_block[f'block_{index}'] = {
                name: values[key].float().square().mean().sqrt().item()
                for name, key in (
                    ('input', 'block_input'),
                    ('after_attention_residual', 'after_attention_residual'),
                    ('after_ffn_residual', 'after_ffn_residual'),
                )
                if key in values
            }
        return rms_by_block
    # --- End added per-block activation RMS ---

    def block_gradient_norms(self):
        squared_norms = {}
        for name, parameter in self._model.named_parameters():
            if parameter.grad is None or not name.startswith('blocks.'):
                continue
            block_index = name.split('.', 2)[1]
            squared_norms[block_index] = (
                squared_norms.get(block_index, 0.)
                + parameter.grad.detach().float().square().sum().item()
            )
        return {
            f'block_{index}': math.sqrt(squared_norms.get(str(index), 0.))
            for index in range(len(self.blocks))
        }

    # --- Added per-block Attention and FFN gradient norms ---
    def block_component_gradient_norms(self):
        norms_by_block = {}
        for index, block in enumerate(self.blocks):
            components = {}
            for name, module_names in (
                ('attention', ('norm1', 'qkv', 'proj')),
                ('ffn', ('norm2', 'mlp')),
            ):
                parameters = {}
                for module_name in module_names:
                    module = getattr(block, module_name, None)
                    if module is not None:
                        parameters.update({id(parameter): parameter for parameter in module.parameters()})
                squared_norm = sum(
                    parameter.grad.detach().float().square().sum().item()
                    for parameter in parameters.values()
                    if parameter.grad is not None
                )
                components[name] = math.sqrt(squared_norm)
            norms_by_block[f'block_{index}'] = components
        return norms_by_block
    # --- End added per-block Attention and FFN gradient norms ---

    def attach_model(self, model):
        self._model = model

    def add_train(self, step, loss, learning_rate, total_gradient_norm,
                  block_gradient_norms, block_component_gradient_norms,
                  residual_update_ratios, block_activation_rms):
        self.rows.append({
            'type': 'train', 'step': step, 'train_loss': loss,
            'learning_rate': learning_rate,
            'total_gradient_norm': float(total_gradient_norm),
            'block_gradient_norms': block_gradient_norms,
            # --- Added detailed block diagnostics ---
            'block_component_gradient_norms': block_component_gradient_norms,
            'residual_update_ratios': residual_update_ratios,
            'block_activation_rms': block_activation_rms,
            # --- End added detailed block diagnostics ---
        })

    def add_validation(self, step, result, final=False):
        row = {
            'type': 'validation', 'step': step, 'bpb': result['bpb'],
            'token_ppl': result['token_ppl'], 'nll_nats': result['nll_nats'],
            'targets': result['targets'],
        }
        if final:
            row['final'] = True
        self.rows.append(row)

    def write(self, path):
        path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))

    def close(self):
        for handle in self._handles:
            handle.remove()