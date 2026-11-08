"""Command-line interfaces for preparation, fitting and RNA-only inference."""
import argparse
import json
from . import workflow

def main(argv=None):
    parser = argparse.ArgumentParser(prog='screlift', description='scReLIFT paper code')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-s', 'prepare-d'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', required=True)
        sub.add_argument('--output', required=True)
    sub = commands.add_parser('train')
    sub.add_argument('--cache', required=True)
    sub.add_argument('--output', required=True)
    sub.add_argument('--seed', type=int, choices=workflow.SEEDS, required=True)
    sub.add_argument('--smoke', action='store_true', help='Two epochs; explicitly excluded from paper summaries')
    sub = commands.add_parser('infer')
    sub.add_argument('--bundle', required=True)
    sub.add_argument('--rna', required=True)
    sub.add_argument('--genes', required=True)
    sub.add_argument('--cell-ids')
    sub.add_argument('--output', required=True)
    sub.add_argument('--device', default='cpu')
    sub = commands.add_parser('evaluate')
    sub.add_argument('--inference', required=True)
    sub.add_argument('--labels', required=True)
    sub.add_argument('--label-cell-ids', required=True)
    sub.add_argument('--fold', required=True)
    sub.add_argument('--output', required=True)
    sub = commands.add_parser('summarize')
    sub.add_argument('--metrics', nargs='+', required=True)
    sub.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare-s':
        result = workflow.prepare_s(args.config, args.output)
    elif args.command == 'prepare-d':
        result = workflow.prepare_d(args.config, args.output)
    elif args.command == 'train':
        result = workflow.train(args.cache, args.output, args.seed, args.smoke)
    elif args.command == 'infer':
        result = workflow.infer(args.bundle, args.rna, args.genes, args.output, args.cell_ids, args.device)
    elif args.command == 'evaluate':
        result = workflow.evaluate(args.inference, args.labels, args.label_cell_ids, args.output, args.fold)
    else:
        result = workflow.summarize(args.metrics, args.output)
    print(json.dumps({'result': str(result) if not isinstance(result, dict) else result}, ensure_ascii=False))

if __name__ == '__main__':
    main()
