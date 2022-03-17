import argparse
import event_warping

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", help="input ES file path")
parser.add_argument("output", help="output PNG file path")
parser.add_argument(
    "--vx",
    type=float,
    default=21.3,
    help="initial X velocity (before optimization) in px/s",
)
parser.add_argument(
    "--vy",
    type=float,
    default=-0.75,
    help="initial Y velocity (before optimization) in px/s",
)
parser.add_argument(
    "--gamma",
    choices=["I", "pow.5", "pow.25"],
    default="pow.5",
    help="gamma ramp function",
)
parser.add_argument(
    "--ratio",
    type=float,
    default=0.01,
    help="remove this ratio of most active pixels",
)
args = parser.parse_args()

width, height, events = event_warping.read_es_file(args.input)
events = event_warping.without_most_active_pixels(events, ratio=args.ratio)

cumulative_map = event_warping.accumulate(
    sensor_size=(width, height),
    events=events,
    velocity=(args.vx / 1e6, args.vy / 1e6),
)

if args.gamma == "I":
    gamma = lambda image: image
elif args.gamma == "pow.5":
    gamma = lambda image: image ** (1 / 2)
elif args.gamma == "pow.25":
    gamma = lambda image: image ** (1 / 4)
else:
    raise Exception(f"unsupported gamma function {args.gamma}")

event_warping.render(
    cumulative_map,
    colormap_name="magma",
    gamma=gamma,
).save(args.output)
