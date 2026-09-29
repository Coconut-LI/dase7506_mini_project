"""Plot the training and validation metrics from training_diagnostics.jsonl."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_diagnostics(path):
    training, validation = [], []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON on line {line_number}: {error}') from error
            if row.get('type') == 'train':
                training.append(row)
            elif row.get('type') == 'validation':
                validation.append(row)
    if not training and not validation:
        raise ValueError(f'No training or validation records found in {path}.')
    return training, validation


def block_keys(records, field):
    keys = set()
    for row in records:
        keys.update(row.get(field, {}))
    return sorted(keys, key=lambda key: int(key.rsplit('_', 1)[-1]))


def style_axis(axis, title, ylabel):
    axis.set_title(title, loc='left', fontweight='bold')
    axis.set_xlabel('Training step')
    axis.set_ylabel(ylabel)
    axis.grid(True, color='#d9e0e5', linewidth=0.8, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)


def show_unavailable(axis):
    axis.text(
        0.5, 0.5, 'Not present in this log; rerun training to collect it.',
        ha='center', va='center', transform=axis.transAxes, color='#697780',
    )


def plot_diagnostics(input_path, output_path):
    training, validation = read_diagnostics(input_path)
    colors = plt.get_cmap('tab10').colors
    figure, axes = plt.subplots(4, 3, figsize=(17, 17), constrained_layout=True)
    figure.suptitle('Training diagnostics', fontsize=17, fontweight='bold')
    (loss_axis, validation_axis, optimization_axis, block_gradient_axis,
     attention_gradient_axis, ffn_gradient_axis, input_rms_axis,
     after_attention_rms_axis, after_ffn_rms_axis, attention_ratio_axis,
     ffn_ratio_axis, unused_axis) = axes.flat
    unused_axis.set_visible(False)

    if training:
        steps = [row['step'] for row in training]
        loss_axis.plot(steps, [row['train_loss'] for row in training], color='#176b87', marker='o', markersize=3)
    style_axis(loss_axis, 'Train loss', 'Cross-entropy loss')

    if validation:
        validation_steps = [row['step'] for row in validation]
        bpb_line, = validation_axis.plot(
            validation_steps, [row['bpb'] for row in validation],
            color='#bc4749', marker='o', label='Validation BPB',
        )
        ppl_axis = validation_axis.twinx()
        ppl_line, = ppl_axis.plot(
            validation_steps, [row['token_ppl'] for row in validation],
            color='#457b55', marker='s', linestyle='--', label='Token PPL',
        )
        validation_axis.legend([bpb_line, ppl_line], ['Validation BPB', 'Token PPL'], frameon=False)
        ppl_axis.set_ylabel('Token PPL')
        ppl_axis.spines['top'].set_visible(False)
        ppl_axis.spines['right'].set_color('#457b55')
        ppl_axis.tick_params(axis='y', colors='#457b55')
    style_axis(validation_axis, 'Validation quality', 'Bits per byte (BPB)')

    if training:
        steps = [row['step'] for row in training]
        lr_line, = optimization_axis.plot(
            steps, [row['learning_rate'] for row in training],
            color='#6a4c93', label='Learning rate',
        )
        total_gradient_axis = optimization_axis.twinx()
        total_grad_line, = total_gradient_axis.plot(
            steps, [row['total_gradient_norm'] for row in training],
            color='#e07a32', label='Total gradient norm',
        )
        optimization_axis.legend([lr_line, total_grad_line], ['Learning rate', 'Total gradient norm'], frameon=False)
        total_gradient_axis.set_ylabel('Total gradient norm')
        total_gradient_axis.spines['top'].set_visible(False)
        total_gradient_axis.spines['right'].set_color('#e07a32')
        total_gradient_axis.tick_params(axis='y', colors='#e07a32')
    style_axis(optimization_axis, 'Learning rate and total gradient', 'Learning rate')

    gradient_keys = block_keys(training, 'block_gradient_norms')
    for index, key in enumerate(gradient_keys):
        block_gradient_axis.plot(
            [row['step'] for row in training],
            [row.get('block_gradient_norms', {}).get(key) for row in training],
            label=key.replace('_', ' '), color=colors[index % len(colors)],
        )
    if gradient_keys:
        block_gradient_axis.legend(frameon=False, ncol=2)
    style_axis(block_gradient_axis, 'Gradient norm by block', 'Pre-clipping gradient norm')

    # --- Added Attention and FFN gradient norm charts ---
    component_keys = block_keys(training, 'block_component_gradient_norms')
    for index, key in enumerate(component_keys):
        color = colors[index % len(colors)]
        for component, axis in (
            ('attention', attention_gradient_axis),
            ('ffn', ffn_gradient_axis),
        ):
            axis.plot(
                [row['step'] for row in training],
                [row.get('block_component_gradient_norms', {}).get(key, {}).get(component)
                 for row in training],
                label=key.replace('_', ' '), color=color,
            )
    for axis, title in (
        (attention_gradient_axis, 'Attention gradient norm by block'),
        (ffn_gradient_axis, 'FFN gradient norm by block'),
    ):
        if component_keys:
            axis.legend(frameon=False, ncol=2)
        else:
            show_unavailable(axis)
        style_axis(axis, title, 'Pre-clipping gradient norm')
    # --- End added Attention and FFN gradient norm charts ---

    # --- Added activation RMS charts ---
    rms_keys = block_keys(training, 'block_activation_rms')
    for index, key in enumerate(rms_keys):
        color = colors[index % len(colors)]
        for stage, axis in (
            ('input', input_rms_axis),
            ('after_attention_residual', after_attention_rms_axis),
            ('after_ffn_residual', after_ffn_rms_axis),
        ):
            axis.plot(
                [row['step'] for row in training],
                [row.get('block_activation_rms', {}).get(key, {}).get(stage)
                 for row in training],
                label=key.replace('_', ' '), color=color,
            )
    for axis, title in (
        (input_rms_axis, 'Block input RMS'),
        (after_attention_rms_axis, 'After Attention residual RMS'),
        (after_ffn_rms_axis, 'After FFN residual RMS'),
    ):
        if rms_keys:
            axis.legend(frameon=False, ncol=2)
        else:
            show_unavailable(axis)
        style_axis(axis, title, 'Activation RMS')
    # --- End added activation RMS charts ---

    ratio_keys = block_keys(training, 'residual_update_ratios')
    for index, key in enumerate(ratio_keys):
        block_ratios = [row.get('residual_update_ratios', {}).get(key, {}) for row in training]
        color = colors[index % len(colors)]
        attention_ratio_axis.plot(
            [row['step'] for row in training],
            [ratios.get('attention') for ratios in block_ratios],
            label=key.replace('_', ' '), color=color,
        )
        ffn_ratio_axis.plot(
            [row['step'] for row in training],
            [ratios.get('ffn') for ratios in block_ratios],
            label=key.replace('_', ' '), color=color,
        )
    if ratio_keys:
        attention_ratio_axis.legend(frameon=False, ncol=2)
        ffn_ratio_axis.legend(frameon=False, ncol=2)
    style_axis(attention_ratio_axis, 'Attention residual update ratio', 'Update norm / residual norm')
    style_axis(ffn_ratio_axis, 'FFN residual update ratio', 'Update norm / residual norm')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor='white')
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path, help='Path to training_diagnostics.jsonl')
    parser.add_argument('--output', type=Path, help='Output image path; defaults beside the input file.')
    args = parser.parse_args()
    output = args.output or args.diagnostics.with_name('training_diagnostics.png')
    print(f'Wrote {plot_diagnostics(args.diagnostics, output)}')


if __name__ == '__main__':
    main()