import argparse
import configuration
import event_warping
import json
import pathlib

dirname = pathlib.Path(__file__).resolve().parent


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--input",
    help="input ES or H5 file path",
    default=str(dirname.parent / "recordings" / configuration.name),
)
parser.add_argument("--output", help="output JSON file path (defaults to stdout)")
parser.add_argument(
    "--method",
    choices=["neldermead", "powell", "lbfgsb", "tnc", "slsqp", "cma"],
    default="lbfgsb",
    help="optimization method, lbfgsb, tnc or neldermead recommended",
)
parser.add_argument(
    "--heuristic",
    choices=["variance", "max"],
    default="variance",
    help="heuristic for estimating contrast",
)
parser.add_argument(
    "--vx",
    type=float,
    default=configuration.velocity[0],
    help="initial X velocity (before optimization) in px/s",
)
parser.add_argument(
    "--vy",
    type=float,
    default=configuration.velocity[1],
    help="initial Y velocity (before optimization) in px/s",
)
parser.add_argument(
    "--ratio",
    type=float,
    default=0.01,
    help="remove this ratio of most active pixels",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    help="initial distribution sigma (cma only)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=20,
    help="number of iterations (cma only)",
)
args = parser.parse_args()


width, height, events = event_warping.read_es_or_h5_file(args.input)
events = event_warping.without_most_active_pixels(events, ratio=args.ratio)


def callback(velocity):
    pass


if args.method == "cma":
    optimized_velocity = event_warping.optimize_cma(
        sensor_size=(width, height),
        events=events,
        initial_velocity=(args.vx / 1e6, args.vy / 1e6),
        heuristic_name=args.heuristic,
        initial_sigma=args.sigma / 1e6,
        iterations=args.iterations,
    )
else:
    optimized_velocity = event_warping.optimize(
        sensor_size=(width, height),
        events=events,
        initial_velocity=(args.vx / 1e6, args.vy / 1e6),
        heuristic_name=args.heuristic,
        method={
            "neldermead": "Nelder-Mead",
            "powell": "Powell",
            "lbfgsb": "L-BFGS-B",
            "tnc": "TNC",
            "slsqp": "SLSQP",
        }[args.method],
        callback=callback,
    )

if args.heuristic == "variance":
    initial_heuristic = event_warping.intensity_variance(
        sensor_size=(width, height),
        events=events,
        velocity=(args.vx / 1e6, args.vy / 1e6),
    )
    optimized_heuristic = event_warping.intensity_variance(
        sensor_size=(width, height),
        events=events,
        velocity=optimized_velocity,
    )
elif args.heuristic == "max":
    initial_heuristic = event_warping.intensity_maximum(
        sensor_size=(width, height),
        events=events,
        velocity=(args.vx / 1e6, args.vy / 1e6),
    )
    optimized_heuristic = event_warping.intensity_maximum(
        sensor_size=(width, height),
        events=events,
        velocity=optimized_velocity,
    )

else:
    raise Exception('unknown heuristic "{args.heuristic}"')

result = json.dumps(
    {
        "method": args.method,
        "heuristic": args.heuristic,
        "initial_vx": args.vx,
        "initial_vy": args.vy,
        "initial_heuristic": initial_heuristic,
        "optimized_vx": optimized_velocity[0] * 1e6,
        "optimized_vy": optimized_velocity[1] * 1e6,
        "optimized_heuristic": optimized_heuristic,
    }
)

if args.output is None:
    print(result)
else:
    with open(args.output, "w") as output_file:
        output_file.write(result)
