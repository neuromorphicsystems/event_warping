from __future__ import annotations
import cmaes
import copy
import dataclasses
import event_stream
import event_warping_extension
import h5py
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import pathlib
import numpy as np
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import scipy.optimize
import typing

def read_es_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, np.ndarray]:
    with event_stream.Decoder(path) as decoder:
        return (
            decoder.width,
            decoder.height,
            np.concatenate([packet for packet in decoder]),
        )


def read_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    data = numpy.asarray(h5py.File(path, "r")["/FalconNeuro"], dtype=numpy.uint32)
    events = numpy.zeros(data.shape[1], dtype=event_stream.dvs_dtype)
    events["t"] = data[3]
    events["x"] = data[0]
    events["y"] = data[1]
    events["on"] = data[2] == 1
    return numpy.max(events["x"].max()) + 1, numpy.max(events["y"]) + 1, events  # type: ignore


def read_es_or_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    if pathlib.Path(path).with_suffix(".es").is_file():
        return read_es_file(path=pathlib.Path(path).with_suffix(".es"))
    elif pathlib.Path(path).with_suffix(".h5").is_file():
        return read_h5_file(path=pathlib.Path(path).with_suffix(".h5"))
    raise Exception(
        f"neither \"{pathlib.Path(path).with_suffix('.es')}\" nor \"{pathlib.Path(path).with_suffix('.h5')}\" exist"
    )


@dataclasses.dataclass
class CumulativeMap:
    pixels: np.ndarray


def without_most_active_pixels(events: np.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = np.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    np.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[
        count[events["x"], events["y"]]
        <= np.percentile(count, 100.0 * (1.0 - ratio))
    ]

def with_most_active_pixels(events: np.ndarray, ratio: float):
    return events[events["x"], events["y"]]

# velocity in px/us
def warp(events: np.ndarray, velocity: tuple[float, float]):
    warped_events = np.array(
        events, dtype=[("t", "<u8"), ("x", "<f8"), ("y", "<f8"), ("on", "?")]
    )
    warped_events["x"] -= velocity[0] * warped_events["t"]
    warped_events["y"] -= velocity[1] * warped_events["t"]
    return warped_events

def unwarp(warped_events: np.ndarray, velocity: tuple[float, float]):
    events = np.zeros(
        len(warped_events),
        dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")],
    )
    events["t"] = warped_events["t"]
    events["x"] = np.round(
        warped_events["x"] + velocity[0] * warped_events["t"]
    ).astype("<u2")
    events["y"] = np.round(
        warped_events["y"] + velocity[1] * warped_events["t"]
    ).astype("<u2")
    events["on"] = warped_events["on"]
    return events



def smooth_histogram(warped_events: np.ndarray):
    return event_warping_extension.smooth_histogram(warped_events)


def accumulate(
    sensor_size: tuple[int, int],
    events: np.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
    )

def render(
    cumulative_map: CumulativeMap,
    colormap_name: str,
    gamma: typing.Callable[[np.ndarray], np.ndarray],
    bounds: typing.Optional[tuple[float, float]] = None,
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name)
    if bounds is None:
        bounds = (cumulative_map.pixels.min(), cumulative_map.pixels.max())
    scaled_pixels = gamma(
        np.clip(
            (cumulative_map.pixels - bounds[0]) / (bounds[1] - bounds[0]),
            0.0,
            1.0,
        )
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8)  # type: ignore
    )
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)


def render_histogram(cumulative_map: CumulativeMap, path: pathlib.Path, title: str):
    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.hist(cumulative_map.pixels.flat, bins=200, log=True)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel("Event count")
    matplotlib.pyplot.ylabel("Pixel count")
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.close()


def intensity_variance(
    sensor_size: tuple[int, int],
    events: np.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_variance(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )
def variance_loss(evmap):
    flating = evmap.flatten()
    res = flating[flating != 0]
    return np.var(res)

def weight_f1(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] = eventmap.pixels[i+1,j+1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 3:
    [i,j] = np.where((x >= vx) & (x <= width) & (y > 0) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < width) & (y > height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y > 0) & (y < ((vy*x)/(vx))))
    eventmap.pixels[i+1,j+1] *= (vy/y[i,j])
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= ((vy*x)/(vx))) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 8:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y < height+vy) & (y > (((vy*(x-width))/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 9:
    [i,j] = np.where((x >= width) & (x <= width+vx) & (y > height) & (y <= (((vy*(x-width))/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 10:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= ((vy*(x-width))/(vx))) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > 0) & (x < vx) & (y > height) & (y <= (((vy*x)/(vx))+height)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*height-vx*y[i,j]+vy*x[i,j])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                = 0
        eventmap.pixels[x < EDGEPX]                         = 0
        eventmap.pixels[y > height+vy-EDGEPX]               = 0
        eventmap.pixels[y < EDGEPX]                         = 0
        eventmap.pixels[y < ((vy*(x-width))/(vx))+EDGEPX]   = 0
        eventmap.pixels[y > (((vy*x)/(vx))+height)-EDGEPX]  = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]              = 0
    return eventmap.pixels

def weight_f2(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y >= (vy*x)/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    upperBase = (vx)/x[i,j]
    # Condition 2:
    [i,j] = np.where((x >= width) & (x < vx) & (y >= vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x >= width) & (x <= vx) & (y >= (vy*x)/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x < vx) & (y >= height) & (y <= (vy/vx)*(x-width-vx)+vy+height))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x <= width) & (y > 0) & (y < (vy*x)/vx))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y >= height) & (y <= (vy/vx)*(x-width-vx)+vy+height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 8:
    [i,j] = np.where((x > 0) & (x < width) & (y > vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i,j]
    # Condition 9:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 10:
    [i,j] = np.where((x > width) & (x < vx) & (y <= (vy*x)/vx) & (y > 0))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x >= vx) & (x <= vx+width) & (y >= (vy*(x-width))/vx) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x > width) & (x < vx) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i-1,j-1]+vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > 0) & (x <= width) & (y < (vy/vx)*x+height) & (y >= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i-1,j-1]+vy*x[i-1,j-1])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
        eventmap.pixels[y < (vy*(x-width))/vx+EDGEPX]                = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f3(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] = eventmap.pixels[i+1,j+1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i-1,j-1]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < width) & (y >= 0) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 4:
    [i,j] = np.where((x > vx) & (x < width) & (y >= height) & (y <= height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 5:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y > height) & (y <= ((-vy*x)/vx)+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx)/x[i-1,j-1]
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= ((-vy*x)/vx)+height+vy) & (y <= height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 8:
    [i,j] = np.where((x > width) & (x < width+vx) & (y >= ((vy*(-x+width+vx))/vx)) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 9:
    [i,j] = np.where((x >= width) & (x < width+vx) & (y >= 0) & (y <= ((vy*(-x+width+vx))/vx)))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 10:
    [i,j] = np.where((x > 0) & (x < vx) & (y > ((-vy*(x))/vx)+vy) & (y < vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x >= width) & (x < width+vx) & (y > height) & (y < ((vy*(-x+width+vx))/vx)+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i,j]+vy*width-vy*x[i,j])
    
    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                        = 0
        eventmap.pixels[x < EDGEPX]                                 = 0
        eventmap.pixels[y > height+vy-EDGEPX]                       = 0
        eventmap.pixels[y < EDGEPX]                                 = 0
        eventmap.pixels[y < (((-vy*(x))/vx)+vy)+EDGEPX]             = 0
        eventmap.pixels[y > ((vy*(-x+width+vx))/vx)+height-EDGEPX]  = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                      = 0
    return eventmap.pixels

def weight_f4(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    upperBase = (vy)/y[i,j]
    # Condition 2:
    [i,j] = np.where((x > vx) & (x < width) & (y >= height) & (y <= vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x < vx) & (y > height) & (y < (vy*x)/vx))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 4:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 5:
    [i,j] = np.where((x > width) & (x < vx+width) & (y <= ((vy*(x-width))/vx)+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i-1,j-1]+width+vx)
    # Condition 6:
    [i,j] = np.where((x >= width) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i-1,j-1]+height+vy)
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y > 0) & (y < (vy*x)/vx) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 8:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (vy*x)/vx) & (y < height))
    eventmap.pixels[i,j] *= (vx)/x[i,j]
    # Condition 9:
    [i,j] = np.where((x >= vx) & (x < width) & (y > vy) & (y <= vy+height))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i-1,j-1]+height+vy)
    # Condition 10:
    [i,j] = np.where((x > 0) & (x <= vx) & (y > (vy/vx)*x) & (y < (vy/vx)*x+height) & (y >= height) & (y < vy))
    eventmap.pixels[i,j] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x >= 0) & (x <= vx) & (y <= (vy/vx)*x+height) & (y > vy) & (y <= height+vy))
    eventmap.pixels[i,j] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x >= width) & (x <= vx+width) & (y <= height) & (y > (vy/vx)*(x-width)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < ((vy*(x-width))/vx)+height) & (y > height) & (y <vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]            = 0
        eventmap.pixels[x < EDGEPX]                     = 0
        eventmap.pixels[y > height+vy-EDGEPX]           = 0
        eventmap.pixels[y < EDGEPX]                     = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]   = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]          = 0
    return eventmap.pixels

def weight_f5(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(x[i,j])
    upperBase = (vx)/(x[i,j])
    # Condition 2:
    [i,j] = np.where((x > width) & (x < vx) & (y > (vy/vx)*(width-x)+vy) & (y < ((-vy*x)/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < width+vx) & (y >= vy) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 4:
    [i,j] = np.where((x > 0) & (x < width) & (y > ((-vy*x)/(vx))+height+vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 5:
    [i,j] =np.where((x > 0) & (x < width) & (y > height) & (y < ((-vy*x)/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] =np.where((x > vx) & (x < vx+width) & (y > (vy*(-x+width+vx))/(vx)) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] =np.where((x > vx) & (x < vx+width) & (y > 0) & (y < (vy*(-x+width+vx))/(vx)))
    eventmap.pixels[i+1,j+1] *= (vy)/y[i,j]
    # Condition 8:
    [i,j] =np.where((x >= width) & (x < vx) & (y > ((-vy*x)/(vx))+height+vy) & (y < ((vy*(-x+width))/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= (vy*vx)/(height*vx+vx*vy-vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 9:
    [i,j] =np.where((x >= vx) & (x < vx+width) & (y > height) & (y < ((vy*(-x+width))/(vx))+height+vy))
    eventmap.pixels[i+1,j+1] *= (vy*vx)/(height*vx+vx*vy-vx*y[i,j]+vy*width-vy*x[i,j])
    # Condition 10:
    [i,j] =np.where((x >= 0) & (x < width) & (y > (-vy/vx)*x+vy) & (y < vy))
    eventmap.pixels[i+1,j+1] *= -(vy*vx)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 11:
    [i,j] =np.where((x >= width) & (x < vx) & (y > (-vy/vx)*x+vy) & (y < (vy/vx)*(width-x)+vy))
    eventmap.pixels[i+1,j+1] *= -(vy*vx)/(vx*vy-vx*y[i,j]-vy*x[i,j])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > ((vy*(-x+width))/(vx))+height+vy-EDGEPX] = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f6(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > vx) & (x < width) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vy)/(y[i,j])
    upperBase = (vy)/(y[i,j])
    # Condition 2:
    [i,j] = np.where((x > vx) & (x < width) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (-vy/vx)*x+vy+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 4:
    [i,j] = np.where((x > width) & (x < vx+width) & (y > height) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < width) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 6:
    [i,j] = np.where((x > 0) & (x < vx) & (y <= (-vy/vx)*x+vy+height) & (y >= (-vy/vx)*x+vy) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 7:
    [i,j] = np.where((x > 0) & (x < vx) & (y >= (-vy/vx)*x+vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 8:
    [i,j] = np.where((x > 0) & (x < vx) & (y > vy) & (y < (-vy/vx)*x+vy+height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 9:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (-vy/vx)*x+vy+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 10:
    [i,j] = np.where((x > width) & (x < vx+width) & (y > (vy/vx)*(-x+width)+vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 11:
    [i,j] = np.where((x > width) & (x < vx+width) & (y > 0) & (y < (vy/vx)*(-x+width)+vy) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 12:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < (vy/vx)*(-x+width)+vy+height) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width)+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < vy+height) & (y > (vy/vx)*(-x+width)+vy) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*(-x+width)+vy+height-EDGEPX]     = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f7(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 4:
    [i,j] = np.where((x > 0) & (x < width) & (y > 0) & (y <= (vy/vx)*x) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    upperBase = vy/y[i,j]
    # Condition 1:
    [i,j] = np.where((x > width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < (vy/vx)*x))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 2:
    [i,j] = np.where((x > 0) & (x <= width) & (y >= height) & (y < (vy/vx)*x))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 4:
    [i,j] = np.where((x > 0) & (x < width) & (y > (vy/vx)*x) & (y <= height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y <= vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width)+height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 8:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 9:
    [i,j] = np.where((x >= width) & (x < vx) & (y > -(vy/vx)*x+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 10:
    [i,j] = np.where((x > 0) & (x < width) & (y > -(vy/vx)*x+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > 0) & (x < vx) & (y > (vy/vx)*x) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x > width) & (x < vx+width) & (y < (vy/vx)*(x-width)+height) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]                = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f8(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y > 0) & (y < (vy/vx)*x) & (y > 0))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    upperBase = vy/y[i,j]
    # Condition 2:
    [i,j] = np.where((x > width) & (x < vx) & (y < (vy/vx)*(x-width)+height) & (y > (vy/vx)*x) & (y > height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > width) & (x < vx) & (y < (vy/vx)*(x-width)+height) & (y > (vy/vx)*x) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 4:
    [i,j] = np.where((x > width) & (x < vx) & (y < (vy/vx)*(x-width)+height) & (y > (vy/vx)*x) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x < width) & (y > (vy/vx)*x) & (y <= height))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width)+height) & (y <= vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width)+height) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 8:
    [i,j] = np.where((x >= width) & (x < vx) & (y < (vy/vx)*x) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 9:
    [i,j] = np.where((x >= width) & (x < vx) & (y < (vy/vx)*x) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 10:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y < (vy/vx)*x) & (y >= 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 11:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y < (vy/vx)*x) & (y >= height) & (y < vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 12:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 13:
    [i,j] = np.where((x >= width) & (x < vx) & (y > (vy/vx)*(x-width)+height) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 14:
    [i,j] = np.where((x >= 0) & (x < width) & (y > (vy/vx)*(x-width)+height) & (y < vy) & (y > height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])
    # Condition 15:
    [i,j] = np.where((x >= 0) & (x < width) & (y > (vy/vx)*(x-width)+height) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx-vx*y[i,j]+vy*x[i,j])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(x-width)+EDGEPX]                = 0
        eventmap.pixels[y > (vy/vx)*x+height-EDGEPX]                 = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def weight_f9(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > 0) & (x < width) & (y > (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    upperBase = (vy)/(-y[i,j]+height+vy)
    # Condition 2:
    [i,j] = np.where((x > width) & (x < vx) & (y > (-vy/vx)*x+height+vy) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= upperBase[0]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x < width) & (y > (-vy/vx)*x+height+vy) & (y < vy))
    eventmap.pixels[i+1,j+1] *= upperBase[0]
    # Condition 4:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > height) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= upperBase[0]
    # Condition 5:
    [i,j] = np.where((x > 0) & (x < width) & (y < (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 6:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y < (vy/vx)*(-x+width+vx)) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 7:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > (vy/vx)*(-x+width+vx)) & (y < height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 8:
    [i,j] = np.where((x > width) & (x < vx) & (y < height) & (y > 0))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 9:
    [i,j] = np.where((x > 0) & (x < width) & (y < height) & (y > 0))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 10:
    [i,j] = np.where((x > 0) & (x < width) & (y > height) & (y < vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 11:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 12:
    [i,j] = np.where((x > width) & (x < vx) & (y > vy) & (y < vy+height))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 14:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 15:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y > (vy/vx)*(-x+width)+vy+height-EDGEPX]     = 0
        eventmap.pixels[y < (-vy/vx)*x+vy+EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                    = 0
    return eventmap.pixels

def weight_f10(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True):
    # Condition 1:
    [i,j] = np.where((x > width) & (x < vx) & (y <= height) & (y > 0) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    upperBase = -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 2:
    [i,j] = np.where((x > width) & (x < vx) & (y < (-vy/vx)*x+height+vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= upperBase[-1]
    # Condition 3:
    [i,j] = np.where((x > 0) & (x <= width) & (y > (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= (vy)/(-y[i,j]+height+vy)
    # Condition 4:
    [i,j] = np.where((x > 0) & (x <= width) & (y < (-vy/vx)*x+height+vy) & (y < vy+height) & (y > vy))
    eventmap.pixels[i+1,j+1] *= vx/x[i,j]
    # Condition 5:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y < (vy/vx)*(-x+width+vx)) & (y > 0) & (y < height))
    eventmap.pixels[i+1,j+1] *= vy/y[i,j]
    # Condition 6:
    [i,j] = np.where((x >= vx) & (x < vx+width) & (y > (vy/vx)*(-x+width+vx)) & (y <= height))
    eventmap.pixels[i+1,j+1] *= (vx)/(-x[i,j]+width+vx)
    # Condition 7:
    [i,j] = np.where((x > 0) & (x <= width) & (y > 0) & (y <= height))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 8:
    [i,j] = np.where((x > 0) & (x <= width) & (y > height) & (y < vy) & (y < (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 9:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y < (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= -(vx*vy)/(vx*vy-vx*y[i,j]-vy*x[i,j])
    # Condition 10:
    [i,j] = np.where((x > width) & (x < vx) & (y > vy) & (y < vy+height) & (y > (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 11:
    [i,j] = np.where((x > width) & (x < vx) & (y > height) & (y < vy) & (y > (-vy/vx)*x+height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 12:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > vy) & (y < height+vy))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])
    # Condition 13:
    [i,j] = np.where((x > vx) & (x < vx+width) & (y > height) & (y < vy) & (y > (vy/vx)*(-x+width+vx)))
    eventmap.pixels[i+1,j+1] *= (vx*vy)/(height*vx+vx*vy-vx*y[i-1,j-1]+vy*width-vy*x[i-1,j-1])

    if remove_edge_pixels:
        eventmap.pixels[x > width+vx-EDGEPX]                         = 0
        eventmap.pixels[x < EDGEPX]                                  = 0
        eventmap.pixels[y > height+vy-EDGEPX]                        = 0
        eventmap.pixels[y < EDGEPX]                                  = 0
        eventmap.pixels[y < (vy/vx)*(-x+width)+vy-EDGEPX]            = 0
        eventmap.pixels[y > (-vy/-x)*x+vy-EDGEPX]                    = 0
    eventmap.pixels[np.isnan(eventmap.pixels)]                       = 0
    return eventmap.pixels

def intensity_weighted_variance(sensor_size: tuple[int, int],events: np.ndarray,velocity: tuple[float, float]):
    np.seterr(divide='ignore', invalid='ignore')
    WEIGHT          = True
    t               = (events["t"][-1]-events["t"][0])/1e6
    EDGEPX          = t
    width           = max(events["x"]+20)
    height          = max(events["y"]+20)
    fieldx          = velocity[0]/1e-6
    fieldy          = velocity[1]/1e-6
    field_velocity  = (fieldx * 1e-6, fieldy * 1e-6)
    eventmap = accumulate((width, height), events, field_velocity)
    vx = np.round(np.abs(fieldx*t))
    vy = np.round(np.abs(fieldy*t))
    x  = np.tile(np.arange(1, eventmap.pixels.shape[1]+1), (eventmap.pixels.shape[0], 1))
    y  = np.tile(np.arange(1, eventmap.pixels.shape[0]+1), (eventmap.pixels.shape[1], 1)).T

    if WEIGHT:
        #f_1(x,y)
        if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height:
            evmap   = weight_f1(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)
            
        #f_2(x,y)
        if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height:
            evmap   = weight_f2(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_3(x,y)
        if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)<=height:
            evmap   = weight_f3(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_4(x,y)
        if fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height or fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f4(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_5(x,y)
        if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)<=height:
            evmap   = weight_f5(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_6(x,y)
        if fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height or fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)<=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f6(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_7(x,y)
        if (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f7(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_8(x,y)
        if  (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (((vy/vx)*width)-height)/(np.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f8(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_9(x,y)
        if (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f9(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)

        #f_10(x,y)
        if  (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height or (height+vy-(vy/vx)*(width+vx))/(np.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and np.abs(fieldx*t)>=width and np.abs(fieldy*t)>=height:
            evmap   = weight_f10(x,y,vx,vy,width,height,eventmap,EDGEPX,remove_edge_pixels=True)
            var     = variance_loss(evmap)
        else:
            var = variance_loss(eventmap.pixels)
    else:
        var = variance_loss(eventmap.pixels)
    return var

def intensity_maximum(
    sensor_size: tuple[int, int],
    events: np.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_maximum(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )

def optimize_local(
    sensor_size: tuple[int, int],
    events: np.ndarray,
    initial_velocity: tuple[float, float],  # px/µs
    heuristic_name: str,  # max or variance
    method: str,  # Nelder-Mead, Powell, L-BFGS-B, TNC, SLSQP
    # see Constrained Minimization in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    callback: typing.Callable[[np.ndarray], None],
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    if method == "Nelder-Mead":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] * 1e3, initial_velocity[1] * 1e3],
            method=method,
            bounds=scipy.optimize.Bounds([-1.0, -1.0], [1.0, 1.0]),
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "BFGS":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    else:
        print("optimisation method is not support.")

def optimize_cma(
    sensor_size: tuple[int, int],
    events: np.ndarray,
    initial_velocity: tuple[float, float],
    initial_sigma: float,
    heuristic_name: str,
    iterations: int,
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unnknown heuristic name "{heuristic_name}"')

    optimizer = cmaes.CMA(
        mean=np.array(initial_velocity) * 1e3,
        sigma=initial_sigma * 1e3,
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
    )
    best_velocity: tuple[float, float] = copy.copy(initial_velocity)
    best_heuristic = np.Infinity
    for _ in range(0, iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = heuristic((x[0] / 1e3, x[1] / 1e3))
            solutions.append((x, value))
        optimizer.tell(solutions)
        velocity_array, heuristic_value = sorted(
            solutions, key=lambda solution: solution[1]
        )[0]
        velocity = (velocity_array[0] / 1e3, velocity_array[1] / 1e3)
        if heuristic_value < best_heuristic:
            best_velocity = velocity
            best_heuristic = heuristic_value
    return (float(best_velocity[0]), float(best_velocity[1]))
