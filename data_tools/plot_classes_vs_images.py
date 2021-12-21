#!/usr/bin/env python3

import argparse, ast, pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def count_images_while_adding_classes(classes_per_img, ordered_classes):
    nums_test_imgs = []
    is_test = [False for i in range(len(classes_per_img))]
    for k in tqdm(ordered_classes):
        for i, ks in enumerate(classes_per_img):
            if not is_test[i] and k in ks:
                is_test[i] = True
        nums_test_imgs.append(sum(is_test))
    return nums_test_imgs


def normalize_index(index, num):
    index = num + index if index < 0 else index
    assert(0 <= index < num)
    return index

def remove_by_indices(lst, inds):
    n = len(lst)
    inds = {normalize_index(i, n) for i in inds}
    return [x for i, x in enumerate(lst) if i not in inds]

def _format_plot_for_axis(a, min=None, max=None, omit_ticks=None,
        label=None, log=False):
    assert(a in {'x', 'y'})

    if min is not None:
        getattr(plt, a + 'lim')(**{'left' if a == 'x' else 'bottom': min})
    if max is not None:
        getattr(plt, a + 'lim')(**{'right' if a == 'x' else 'top': max})

    ts = list(getattr(plt, a + 'ticks')()[0])
    if min is not None:
        ts = [min] + [t for t in ts if t > min]
    if max is not None:
        ts = [t for t in ts if t < max] + [max]
    if omit_ticks is not None:
        ts = remove_by_indices(ts, omit_ticks)
    getattr(plt, a + 'ticks')(ts)

    if label is not None:
        getattr(plt, a + 'label')(label)

    if log:
        getattr(plt, a + 'scale')('log')
    
def format_plot(xmin=None, xmax=None, ymin=None, ymax=None,
        omit_xticks=None, omit_yticks=None,
        xlabel=None, ylabel=None, xlog=False, ylog=False):
    _format_plot_for_axis('x', min=xmin, max=xmax, omit_ticks=omit_xticks,
            label=xlabel, log=xlog)
    _format_plot_for_axis('y', min=ymin, max=ymax, omit_ticks=omit_yticks,
            label=ylabel, log=ylog)


def int_tuple(s):
    return tuple(int(x) for x in ast.literal_eval(s))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='infreq')
    parser.add_argument('-t', '--trials', type=int, default=10)
    parser.add_argument('--omit-last-xtick', action='store_true')
    parser.add_argument('--omit-last-ytick', action='store_true')
    parser.add_argument('--figsize', type=int_tuple)
    parser.add_argument('--dpi', type=int) 
    parser.add_argument('-O', '--output')
    parser.add_argument('stat')
    args = parser.parse_args()

    with open(args.stat, 'rb') as f:
        stat = pickle.load(f)
    nonunknown_classes = stat['known_classes'] | stat['missing_classes']
    ordered_classes = [k for k in stat['ordered_classes']
            if k not in nonunknown_classes] # unknown classes, frequent first
    classes_per_img = stat['classes_per_img']
    if args.mode == 'rand':
        results = []
        for t in range(args.trials):
            classes = ordered_classes.copy()
            np.random.shuffle(classes)
            results.append(
                    count_images_while_adding_classes(classes_per_img, classes))
        result = np.stack(results, axis=0)
        means = result.mean(axis=0) # nums. images per added class
        stds  = result.std (axis=0)
    else:
        if args.mode == 'freq':
            classes = ordered_classes # frequent first
        elif args.mode == 'infreq':
            classes = ordered_classes[::-1] # infrequent first
        else:
            raise ValueError(f"invalid mode: '{args.mode}'")
        means = np.asarray(
                count_images_while_adding_classes(classes_per_img, classes))
        stds  = None

    plt.figure(figsize=args.figsize, dpi=args.dpi)
    points = list(range(1, len(classes) + 1)) # nums. of added classes
    plt.errorbar(points, means, stds, ecolor='lightgray')
    format_plot(xmin=1, xmax=len(ordered_classes),
            ymin=0, ymax=len(classes_per_img),
            omit_xticks=[-2] if args.omit_last_xtick else None,
            omit_yticks=[-2] if args.omit_last_ytick else None,
            xlabel='The number of unknown classes', ylabel='The total number of testing images')

    plt.tight_layout()
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)

if __name__ == '__main__':
    main()
